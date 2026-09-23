"""Enforce the §16 repository dependency rule mechanically.

Imports may flow only:
    contracts → core → estimation → {controller, metrics} → components → pipeline

A component importing another component, or core importing a component, is a
build failure. This is what makes the shared-service guarantees (§IF-06) and the
component decoupling (§11) structural rather than aspirational.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "r2r_control"

# layer rank: a module may import only from layers with rank <= its own
LAYER_RANK = {
    "contracts": 0,
    "config": 1,
    "core": 2,
    "estimation": 3,
    "controller": 4,
    "metrics": 4,
    "components": 5,
    "pipeline": 6,
}


def _layer_of(parts):
    if not parts:
        return None, None
    layer = parts[0]
    comp = parts[1] if layer == "components" and len(parts) > 1 else None
    return layer, comp


def _resolve(module_parts, node):
    """Resolve an ImportFrom to absolute parts under r2r_control, or None."""
    if node.level == 0:
        if node.module and node.module.startswith("r2r_control."):
            return node.module.split(".")[1:]
        return None
    # relative import: level 1 = the importing module's package, level 2 = its parent...
    if node.level - 1 > len(module_parts):
        return None
    base = module_parts[:len(module_parts) - (node.level - 1)]
    return base + (node.module.split(".") if node.module else [])


def _iter_modules():
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).with_suffix("")
        parts = list(rel.parts)
        is_pkg = parts[-1] == "__init__"
        if is_pkg:
            parts = parts[:-1]
        # package path of the module (for resolving relative imports)
        pkg = parts if is_pkg else parts[:-1]
        yield path, parts, pkg


def test_layer_dependency_order():
    assert SRC.exists(), "src/r2r_control must exist"
    violations = []
    for path, parts, pkg in _iter_modules():
        src_layer, src_comp = _layer_of(parts)
        if src_layer not in LAYER_RANK:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.ImportFrom):
                t = _resolve(pkg, node)
                if t is not None:
                    targets.append(t)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("r2r_control."):
                        targets.append(alias.name.split(".")[1:])
            for tgt in targets:
                tgt_layer, tgt_comp = _layer_of(tgt)
                if tgt_layer not in LAYER_RANK:
                    continue
                if LAYER_RANK[tgt_layer] > LAYER_RANK[src_layer]:
                    violations.append(f"{'/'.join(parts)} ({src_layer}) imports "
                                      f"{'/'.join(tgt)} ({tgt_layer}) — violates layer order")
                if (src_layer == "components" and tgt_layer == "components"
                        and src_comp and tgt_comp and src_comp != tgt_comp):
                    violations.append(f"component '{src_comp}' imports component "
                                      f"'{tgt_comp}' — components must not depend on "
                                      f"each other (§11)")
    assert not violations, "dependency-rule violations:\n" + "\n".join(violations)


def test_old_package_removed():
    old = SRC.parents[1] / "process_control"
    assert not old.exists(), "legacy process_control/ package should be removed"

from __future__ import annotations

import importlib
import platform
import sys


PACKAGES=[
    ("numpy","numpy"),
    ("pytest","pytest"),
    ("scipy","scipy"),
    ("gurobipy","gurobipy"),
]


def package_version(module_name: str) -> tuple[bool,str]:
    try:
        module=importlib.import_module(module_name)
    except ImportError:
        return False,"not installed"

    version=getattr(module,"__version__",None)
    if version is None and module_name=="gurobipy":
        try:
            version=".".join(str(part) for part in module.gurobi.version())
        except Exception:
            version="installed, version unknown"
    if version is None:
        version="installed, version unknown"
    return True,str(version)


def main() -> None:
    print("Python environment")
    print(f"  executable: {sys.executable}")
    print(f"  version: {platform.python_version()}")
    print(f"  platform: {platform.platform()}")
    print()
    print("Packages")
    for display_name,module_name in PACKAGES:
        installed,version=package_version(module_name)
        status="installed" if installed else "missing"
        print(f"  {display_name:<9} {status:<9} {version}")


if __name__=="__main__":
    main()

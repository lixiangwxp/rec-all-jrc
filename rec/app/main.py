from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rec.config import PROJECT_ROOT, SERVICE_MODULES, resolve_service_name
from rec.pipeline import cli_main


def list_services() -> None:
    print("Available services:")
    for service_name in sorted(SERVICE_MODULES):
        print(f"  {service_name}")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--list" in args:
        list_services()
        return
    if args and not args[0].startswith("-"):
        service_name = resolve_service_name(args[0])
        if service_name not in SERVICE_MODULES:
            print(f"Unknown service {args[0]!r}. Run `python -m rec.app.main --list` from {PROJECT_ROOT}.")
            return
        args = args[1:]
    cli_main(args)


if __name__ == "__main__":
    main()

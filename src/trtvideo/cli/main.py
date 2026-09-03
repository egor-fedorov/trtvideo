"""Backward-compatible command dispatcher for the trtvideo executable."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int | None:
    """Dispatch environment diagnostics or preserve the processing CLI."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "doctor":
        from trtvideo.cli.doctor import main as doctor_main

        return doctor_main(arguments[1:])
    if arguments and arguments[0] == "compatibility-report":
        from trtvideo.cli.compatibility_report import main as compatibility_report_main

        return compatibility_report_main(arguments[1:])
    if arguments and arguments[0] == "compatibility-check":
        from trtvideo.cli.compatibility_check import main as compatibility_check_main

        return compatibility_check_main(arguments[1:])

    from trtvideo.cli.process import main as process_main

    process_main(arguments)
    return None


if __name__ == "__main__":
    raise SystemExit(main())

"""Enable ``python -m spatialaim ...`` as an alias for the ``spatialaim`` console script."""

from spatialaim.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

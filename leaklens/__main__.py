"""Enable `python -m leaklens`."""
import sys

from leaklens.cli import main

if __name__ == "__main__":
    sys.exit(main())

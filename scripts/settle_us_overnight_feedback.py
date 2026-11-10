"""Compatibility entry point for the versioned open-to-open paper settle workflow."""

from run_us_open2open import main


if __name__ == "__main__":
    raise SystemExit(main("settle"))

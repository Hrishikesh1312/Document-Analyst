from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    entrypoint = Path(__file__).resolve().with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(entrypoint)])


if __name__ == "__main__":
    raise SystemExit(main())

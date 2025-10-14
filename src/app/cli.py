"""CLI helpers for launching the Rag'it Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path
from streamlit.web.cli import main as streamlit_main

def main() -> None:
    """Invoke Streamlit with the packaged app entrypoint."""


    app_path = Path(__file__).with_name("main.py")
    sys.argv = ["streamlit", "run", str(app_path)]
    streamlit_main()

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local Streamlit review UI.")
    parser.add_argument("--results_dir", required=True, help="Single run directory or batch results root.")
    parser.add_argument("--server_port", type=int, default=8501, help="Streamlit server port.")
    parser.add_argument("--server_address", default="127.0.0.1", help="Streamlit server address.")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit in headless mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_path = Path(__file__).with_name("review_ui.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.server_port),
        "--server.address",
        str(args.server_address),
        "--server.headless",
        "true" if args.headless else "false",
        "--",
        "--results_dir",
        args.results_dir,
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()

import subprocess
import sys
from pathlib import Path


def main() -> int:
    train_script = Path(__file__).with_name("train_savlg_imagenet_standalone.py")
    command = [sys.executable, str(train_script), "--mode", "precompute_targets", *sys.argv[1:]]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())

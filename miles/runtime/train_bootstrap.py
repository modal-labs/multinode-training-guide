import runpy
import sys

from transformers_compat import register_transformers_compat


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: train_bootstrap.py /path/to/train.py [args...]")

    train_script = sys.argv[1]
    register_transformers_compat()
    sys.argv = [train_script, *sys.argv[2:]]
    runpy.run_path(train_script, run_name="__main__")


if __name__ == "__main__":
    main()

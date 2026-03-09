import argparse

from .config import load_config
from .algorithm import run_dak_ucb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = run_dak_ucb(cfg)
    print(f"\nWrote results to: {out}")


if __name__ == "__main__":
    main()

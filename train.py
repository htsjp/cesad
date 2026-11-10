"""Train only the final CES direction on an explicitly prepared source pool."""

import argparse
from cesad.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    from cesad.training import run

    run(load_config(a.config), a.output, a.device, a.dry_run)


if __name__ == "__main__":
    main()

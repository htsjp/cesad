"""Evaluate saved anomaly maps against original paired GT; no model inference."""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    from cesad.evaluation import run

    run(a.manifest, a.output)


if __name__ == "__main__":
    main()

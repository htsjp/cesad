import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("missing command")

    root = Path(__file__).resolve().parents[1]
    stdout_path = root / args.stdout
    stderr_path = root / args.stderr
    pid_path = root / args.pid_file
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP", "CREATE_NO_WINDOW"):
        flags |= getattr(subprocess, name, 0)
    with open(stdout_path, "ab", buffering=0) as out, open(stderr_path, "ab", buffering=0) as err:
        proc = subprocess.Popen(args.command, cwd=root, stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=flags)
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    print(proc.pid)


if __name__ == "__main__":
    main()

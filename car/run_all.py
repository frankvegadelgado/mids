"""run_all -- runs every car experiment script of Siriaisa, one after another.

The scripts of this folder are independent: each one builds its own instances,
runs ``siriaisa`` against an exact SciPy optimum and writes its CSV or JSON
files under ``car/results``. This runner calls them in the order of the paper's
experiment sections, with the default arguments of each script, or with the
quick arguments when ``--quick`` is given.

Each script is started as a separate process from the ``car`` folder, and it
adds the repository's own ``siriaisa`` package to ``sys.path``, so the runner
works whether or not the package is installed.

Usage
-----
    python run_all.py                       # every script, default sizes
    python run_all.py --quick               # every script, small sizes
    python run_all.py --only car_bipartite car_ugc
    python run_all.py --skip run_car
    python run_all.py --keep-going          # continue after a failing script

Outputs
-------
    the console                       the output of each script, as it is produced
    results/run_all/<script>.log      the same output, one file per script
    results/run_all_summary.csv       one row per script: name, status, exit code,
                                      seconds, log file
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

CAR_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CAR_DIR / "results"
LOG_DIR = RESULTS_DIR / "run_all"

# name -> (default arguments, quick arguments), in the order of the paper's sections.
SCRIPTS: dict[str, tuple[list[str], list[str]]] = {
    "run_car": ([], ["--count", "200"]),
    "car_bounded": ([], ["--quick"]),
    "car_bipartite": ([], ["--quick"]),
    "car_circle": ([], ["--quick"]),
    "car_cubic": ([], ["--quick"]),
    "car_split": ([], ["--quick"]),
    "car_ugc": ([], ["--quick"]),
}


def run_script(name: str, extra: list[str], keep_going: bool) -> dict:
    """Runs one script from the car folder; its output is shown on the console as it is
    produced and is also written to its log file. Returns its summary row."""
    log_path = LOG_DIR / f"{name}.log"
    command = [sys.executable, "-u", f"{name}.py", *extra]
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    started = time.time()
    print(f"[run_all] {name}: {' '.join(command)}", flush=True)
    console = sys.stdout.buffer
    with open(log_path, "wb") as log:
        log.write((" ".join(command) + "\n\n").encode("utf-8"))
        process = subprocess.Popen(command, cwd=CAR_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=environment)
        # Copy the output in small chunks, not lines, so progress lines ending in "\r" appear at once.
        while True:
            chunk = process.stdout.read1(4096)
            if not chunk:
                break
            console.write(chunk)
            console.flush()
            log.write(chunk)
            log.flush()
        process.wait()
    seconds = time.time() - started
    status = "ok" if process.returncode == 0 else "failed"
    print(f"[run_all] {name}: {status} (exit {process.returncode}) in {seconds:.1f} s, log {log_path}", flush=True)
    if process.returncode != 0 and not keep_going:
        raise SystemExit(f"run_all stopped: {name} exited with {process.returncode} (see {log_path})")
    return {"script": name, "status": status, "exit_code": process.returncode,
            "seconds": round(seconds, 3), "log": str(log_path.relative_to(CAR_DIR))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="run every script with its small-size arguments")
    parser.add_argument("--only", nargs="+", choices=list(SCRIPTS), metavar="SCRIPT",
                        help="run only these scripts (in the usual order)")
    parser.add_argument("--skip", nargs="+", choices=list(SCRIPTS), metavar="SCRIPT", default=[],
                        help="do not run these scripts")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue with the next script when one fails (default: stop)")
    args = parser.parse_args()

    selected = [name for name in SCRIPTS if (args.only is None or name in args.only) and name not in args.skip]
    if not selected:
        raise SystemExit("run_all: no script selected")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for name in selected:
        default_args, quick_args = SCRIPTS[name]
        extra = quick_args if args.quick else default_args
        rows.append(run_script(name, extra, args.keep_going))

    summary = RESULTS_DIR / "run_all_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["script", "status", "exit_code", "seconds", "log"])
        writer.writeheader()
        writer.writerows(rows)

    failed = [row["script"] for row in rows if row["status"] != "ok"]
    print(f"[run_all] {len(rows) - len(failed)}/{len(rows)} scripts ok in {time.time() - started:.1f} s; "
          f"summary {summary}", flush=True)
    if failed:
        print(f"[run_all] failed: {', '.join(failed)}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

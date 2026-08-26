# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import inspect
import os
import re
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path

from monai.utils import PerfContext

results: dict = {}

_SIGALRM_AVAILABLE = hasattr(signal, "SIGALRM")


class _TestTimeoutError(Exception):
    """Raised when a single test exceeds the per-test timeout."""


def _alarm_handler(signum, frame):
    raise _TestTimeoutError("Test timed out")


class TimeLoggingTestResult(unittest.TextTestResult):
    """Overload the default results so that we can store the results."""

    # Set by the caller before running; 0 means no timeout.
    timeout: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timed_tests = {}

    def startTest(self, test):  # noqa: N802
        """Start timer, print test name, do normal test."""
        self.start_time = time.time()
        name = self.getDescription(test)
        self.stream.write(f"Starting test: {name}...\n")
        if _SIGALRM_AVAILABLE and self.timeout > 0:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(self.timeout)
        super().startTest(test)

    def stopTest(self, test):  # noqa: N802
        """On test end, get time, print, store and do normal behaviour."""
        if _SIGALRM_AVAILABLE and self.timeout > 0:
            signal.alarm(0)  # cancel any pending alarm
        elapsed = time.time() - self.start_time
        name = self.getDescription(test)
        self.stream.write(f"Finished test: {name} ({elapsed:.03}s)\n")
        if name in results:
            raise AssertionError(f"expected all keys to be unique, but {name} is duplicated")
        results[name] = elapsed
        super().stopTest(test)


def print_results(results, discovery_time, thresh, status):
    # only keep results >= threshold
    results = dict(filter(lambda x: x[1] > thresh, results.items()))
    if len(results) == 0:
        return
    print(f"\n\n{status}, printing completed times >{thresh}s in ascending order...\n")
    timings = dict(sorted(results.items(), key=lambda item: item[1]))

    for r in timings:
        if timings[r] >= thresh:
            print(f"{r} ({timings[r]:.03}s)")
    print(f"test discovery time: {discovery_time:.03}s")
    print(f"total testing time: {sum(results.values()):.03}s")
    print("Remember to check above times for any errors!")


def parse_args():
    parser = argparse.ArgumentParser(description="Runner for MONAI unittests with timing.")
    parser.add_argument(
        "-s", action="store", dest="path", default=".", help="Directory to start discovery (default: '%(default)s')"
    )
    parser.add_argument(
        "-p",
        action="store",
        dest="pattern",
        default="test_*.py",
        help="Pattern to match tests (default: '%(default)s')",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        dest="thresh",
        default=10.0,
        type=float,
        help="Display tests longer than given threshold (default: %(default)d)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store",
        dest="verbosity",
        type=int,
        default=1,
        help="Verbosity level (default: %(default)d)",
    )
    parser.add_argument("-q", "--quick", action="store_true", dest="quick", default=False, help="Only do quick tests")
    parser.add_argument(
        "-f", "--failfast", action="store_true", dest="failfast", default=False, help="Stop testing on first failure"
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        default=0,
        type=int,
        help="Per-test timeout in seconds; 0 disables (default: %(default)d). Requires SIGALRM (Linux/macOS only).",
    )
    _rerun_default_env = os.environ.get("MONAI_TEST_RERUN_FAILURES", "0")
    try:
        _rerun_default = int(_rerun_default_env)
    except ValueError:
        print(f"Warning: invalid MONAI_TEST_RERUN_FAILURES={_rerun_default_env!r}; defaulting to 0.")
        _rerun_default = 0
    parser.add_argument(
        "--rerun-failures",
        dest="rerun_failures",
        default=_rerun_default,
        type=int,
        help="Re-run each failed/errored test in a FRESH subprocess up to N times; the suite "
        "passes if they recover (default: %(default)d, or env MONAI_TEST_RERUN_FAILURES). "
        "Mitigates the intermittent ROCm/MIOpen kernel code-object load fault "
        "('device kernel image is invalid' / 'named symbol not found' -> miopenStatusUnknownError) "
        "that hits a few random tests per full suite run on gfx942. A fresh process clears the "
        "HIP context (in-process retry does NOT recover it); a genuine failure keeps failing every "
        "attempt and is still reported. "
        "Each retry subprocess is bounded by --timeout if set, otherwise by a 300s wall-clock "
        "cap (independent of --timeout 0) to prevent a hanging test from blocking the retry loop.",
    )
    # Internal flag: when set, run exactly the given test ids and exit (used by the rerun step).
    parser.add_argument("--only-ids", dest="only_ids", nargs="+", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.only_ids:  # suppress discovery noise in per-test worker subprocesses
        print(f"Running tests in folder: '{args.path}'")
        if args.pattern:
            print(f"With file pattern: '{args.pattern}'")

    return args


def get_default_pattern(loader):
    signature = inspect.signature(loader.discover)
    params = {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
    return params["pattern"]


def rerun_failed_tests_fresh(failed_ids, attempts, verbosity, timeout=0):
    """Re-run each failed/errored test id in a FRESH subprocess, up to ``attempts`` times.

    A fresh Python process gives a clean HIP/MIOpen context, which is required to clear the
    intermittent 'device kernel image is invalid' load fault (proven: in-process retry does
    NOT recover it, but a new process does). Returns the list of ids still failing after all
    retries; a genuine failure keeps failing every attempt and is returned.

    Ids that represent fixture-level errors (_ErrorHolder, e.g. setUpClass/setUpModule) are
    skipped — they cannot be loaded by name and are returned as permanently failing.
    """
    _script = str(Path(__file__).resolve())
    _monai_root = str(Path(__file__).resolve().parent.parent)

    # Filter out _ErrorHolder ids (setUpClass/setUpModule) that loadTestsFromNames cannot resolve.
    rerunnable = [tid for tid in failed_ids if not tid.startswith(("setUpClass", "setUpModule"))]
    non_rerunnable = [tid for tid in failed_ids if tid not in rerunnable]
    if non_rerunnable:
        print(f"  skipping {len(non_rerunnable)} fixture-level error(s) (cannot rerun by id): "
              + ", ".join(non_rerunnable))

    still_failing = list(rerunnable)
    for attempt in range(1, attempts + 1):
        if not still_failing:
            break
        print(f"\n{'=' * 70}")
        print(f"Re-running {len(still_failing)} failed/errored test(s) in fresh subprocess(es), "
              f"attempt {attempt}/{attempts}")
        for tid in still_failing:
            print(f"  retry: {tid}")
        print("=" * 70)
        recovered = []
        for tid in list(still_failing):
            # Each test id in its own fresh process -> clean HIP context.
            # Use resolved script path so cwd change does not break relative __file__.
            # Wall-clock limit per retry: honour explicit --timeout if set, else cap at 5 min
            # so a hanging test never blocks the retry loop indefinitely.
            _per_test_timeout = timeout if timeout > 0 else 300
            cmd = [sys.executable, _script, "--only-ids", tid, "--verbosity", str(verbosity)]
            if timeout > 0:
                cmd += ["--timeout", str(timeout)]
            try:
                rc = subprocess.run(cmd, cwd=_monai_root, timeout=_per_test_timeout).returncode
            except subprocess.TimeoutExpired:
                print(f"  retry timed out after {_per_test_timeout}s: {tid}")
                rc = 1
            if rc == 0:
                print(f"  recovered on retry: {tid}")
                recovered.append(tid)
        still_failing = [tid for tid in still_failing if tid not in recovered]

    return still_failing + non_rerunnable


if __name__ == "__main__":
    # Run as a script, sys.path[0] is this file's dir (tests/), which breaks `import tests.*`
    # test ids. Ensure the monai root (parent of tests/) is importable in every mode.
    _monai_root = str(Path(__file__).resolve().parent.parent)
    if _monai_root not in sys.path:
        sys.path.insert(0, _monai_root)

    # Parse input arguments
    args = parse_args()

    # If quick is desired, set environment variable
    if args.quick:
        os.environ["QUICKTEST"] = "True"

    # ------------------------------------------------------------------ #
    # WORKER MODE: run exactly the given test id(s) in this fresh process #
    # and exit. Used by the --rerun-failures step so each retried test    #
    # gets a clean HIP/MIOpen context.                                    #
    # ------------------------------------------------------------------ #
    if args.only_ids:
        # As a script, sys.path[0] is this file's dir (tests/), so `import tests` fails.
        # Put the monai root (parent of tests/) on the path so test ids resolve.
        monai_root = str(Path(__file__).resolve().parent.parent)
        if monai_root not in sys.path:
            sys.path.insert(0, monai_root)
        rerun_suite = unittest.TestLoader().loadTestsFromNames(args.only_ids)
        rerun_result = unittest.runner.TextTestRunner(verbosity=args.verbosity).run(rerun_suite)
        sys.exit(0 if rerun_result.wasSuccessful() else 1)

    # Get all test names (optionally from some path with some pattern)
    with PerfContext() as pc:
        # the files are searched from `tests/` folder, starting with `test_`
        tests_path = Path(__file__).parent / args.path
        files = {
            file.relative_to(tests_path).as_posix()
            for file in tests_path.rglob("test_*py")
            if re.search(args.pattern, file.name[:-3])
        }
        print(files)
        cases = []
        for test_module in tests_path.rglob("test_*py"):
            test_file = str(test_module.relative_to(tests_path).as_posix())
            case_str = test_file.replace("/", ".")[:-3]
            case_str = f"tests.{case_str}"
            if test_file in files:
                cases.append(case_str)
            else:
                print(f"monai test runner: excluding {test_module.name}")
        print(cases)
        tests = unittest.TestLoader().loadTestsFromNames(cases)
    discovery_time = pc.total_time
    print(f"time to discover tests: {discovery_time}s, total cases: {tests.countTestCases()}.")

    if args.timeout > 0:
        if _SIGALRM_AVAILABLE:
            TimeLoggingTestResult.timeout = args.timeout
            print(f"Per-test timeout enabled: {args.timeout}s")
        else:
            print("Warning: --timeout ignored; SIGALRM is not available on this platform.")

    test_runner = unittest.runner.TextTestRunner(
        resultclass=TimeLoggingTestResult, verbosity=args.verbosity, failfast=args.failfast
    )
    # Use try catches to print the current results if encountering exception or keyboard interruption
    try:
        test_result = test_runner.run(tests)
        print_results(results, discovery_time, args.thresh, "tests finished")
        if test_result.wasSuccessful():
            sys.exit(0)
        if args.rerun_failures > 0 and not args.failfast:
            # unexpectedSuccesses are not in errors/failures but do make wasSuccessful() False.
            # Treat them as non-recoverable so they are never masked.
            unexpected = getattr(test_result, "unexpectedSuccesses", [])
            if unexpected:
                print(f"\n{len(unexpected)} unexpected success(es) — not retried:")
                for t in unexpected:
                    print(f"  UNEXPECTED SUCCESS: {t.id()}")
                sys.exit(1)
            failed_ids = [t.id() for t, _ in list(test_result.errors) + list(test_result.failures)]
            still_failing = rerun_failed_tests_fresh(
                failed_ids, args.rerun_failures, args.verbosity, timeout=args.timeout
            )
            if not still_failing:
                print(
                    f"\nAll initially failed/errored test(s) passed within {args.rerun_failures} "
                    "fresh-process retry/ies; treating as transient. Suite result: SUCCESS."
                )
                sys.exit(0)
            print(f"\n{len(still_failing)} test(s) still failing after {args.rerun_failures} retry/ies:")
            for tid in still_failing:
                print(f"  FAILED: {tid}")
            sys.exit(1)
        sys.exit(not test_result.wasSuccessful())
    except KeyboardInterrupt:
        print_results(results, discovery_time, args.thresh, "tests cancelled")
        sys.exit(1)
    except Exception:
        print_results(results, discovery_time, args.thresh, "exception reached")
        raise

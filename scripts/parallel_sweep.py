'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Run the planned three-scenario date sweep with one worker process per
(scenario, date) task. Usage from the repository root:

    python3 -m scripts.parallel_sweep --root output/frozen [--workers 10]
        [--scenarios 1 2 3] [--dates 20251201 ...] [--max-tasks N]
        [--overwrite] [--dry-run]

Each task writes to <root>/s<scenario>_<date>/, matching the layout expected
by scripts.aggregate_sweep. Dates whose summary.json already holds a terminal
status are skipped (resume); --overwrite reruns them. Per-task results are
appended to <root>/sweep_log.jsonl so a crash or interruption stays
distinguishable from a date that never started. --max-tasks caps how many
pending dates actually run this invocation (for timing calibration before the
full sweep); because the task list is longest-scenario-first, --max-tasks 3
samples one date from each scenario.
'''
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# BLAS thread caps must be set before numpy is first imported; otherwise each
# worker's linear algebra would spawn its own thread pool and oversubscribe
# the machine. Forked worker processes inherit the parent's capped state.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

from scripts.aggregate_sweep import RANGES

# The three date-level terminal statuses of the evaluation protocol. Anything
# else (missing file, malformed JSON, in-progress run) is treated as pending
# and will be rerun rather than silently skipped.
TERMINAL_STATUSES = ('success', 'failed', 'unresolved')


def run_one(scenario, day, root):
    """Execute one scenario-date search and return a JSON-serializable record.

    src is imported inside the worker so that forking shares the parent's
    already thread-capped BLAS state instead of re-importing uncapped modules.
    All exceptions are captured into the record: one bad date must not abort
    the remaining sweep.
    """
    from src.inputs import load_system_config
    from src.run import execute
    started = time.perf_counter()
    record = {'scenario': scenario, 'date': day}
    try:
        config = load_system_config('subject.json', 'scenario_profiles.json',
                                    'components.json', scenario, day)
        summary = execute(config, 'search',
                          Path(root) / f's{scenario}_{day}',
                          progress=lambda message: None)
        record.update(status=summary['status'], selected=summary['selected'],
                      counts=summary['counts'])
    except Exception as error:  # noqa: BLE001 - record and continue the sweep
        record.update(status='error',
                      error=f'{type(error).__name__}: {str(error)[:200]}')
    record['wall_seconds'] = round(time.perf_counter() - started, 2)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Output root, e.g. output/frozen_20260929.')
    parser.add_argument('--workers', type=int, default=10,
                        help='Worker processes (default 10).')
    parser.add_argument('--scenarios', type=int, nargs='+', choices=(1, 2, 3),
                        default=(1, 2, 3))
    parser.add_argument('--dates', nargs='+', default=None,
                        help='Optional explicit YYYYMMDD subset within the planned ranges.')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Run at most N pending dates (timing calibration). '
                             'Task order is longest-scenario-first, so N=3 samples one date per scenario.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rerun dates that already have a terminal summary.json.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the task list and exit without computing.')
    args = parser.parse_args(argv)

    # Longest scenario first keeps the pool packed: slow s3 dates start
    # immediately instead of trailing the fast s1 dates at the end.
    selected = set(args.dates) if args.dates else None
    tasks = [(scenario, day)
             for scenario in sorted(args.scenarios, key=lambda s: -len(RANGES[s]))
             for day in RANGES[scenario]
             if selected is None or day in selected]
    if args.dry_run:
        shown = tasks if args.max_tasks is None else tasks[:args.max_tasks]
        print(f'{len(shown)} of {len(tasks)} tasks -> {args.root}')
        for scenario, day in shown:
            print(f'  s{scenario}_{day}')
        return

    pending = []
    for scenario, day in tasks:
        path = args.root / f's{scenario}_{day}' / 'summary.json'
        try:
            status = json.loads(path.read_text(encoding='utf-8')).get('status')
        except (OSError, json.JSONDecodeError):
            status = None
        if status in TERMINAL_STATUSES and not args.overwrite:
            print(f's{scenario}_{day} skip (already {status})', flush=True)
        else:
            pending.append((scenario, day))
    if args.max_tasks is not None:
        pending = pending[:args.max_tasks]
    print(f'{len(pending)} of {len(tasks)} dates to run on '
          f'{args.workers} workers -> {args.root}', flush=True)

    args.root.mkdir(parents=True, exist_ok=True)
    counts = {}
    started_all = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            (args.root / 'sweep_log.jsonl').open('a', encoding='utf-8') as log:
        futures = [pool.submit(run_one, scenario, day, str(args.root))
                   for scenario, day in pending]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            counts[record['status']] = counts.get(record['status'], 0) + 1
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(f"[{done}/{len(pending)}] s{record['scenario']}_{record['date']} "
                  f"status={record['status']} "
                  f"selected={record.get('selected')} "
                  f"wall={record.get('wall_seconds', 0):.0f}s", flush=True)
    print(f'sweep done in {(time.perf_counter() - started_all) / 3600:.2f} h; '
          f'statuses: {counts}', flush=True)
    print(f'next: python3 -m scripts.aggregate_sweep --input_root {args.root} '
          f'--output_dir {args.root}/date_report', flush=True)


if __name__ == '__main__':
    main()

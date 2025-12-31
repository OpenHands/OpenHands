import argparse
import fnmatch
import json
from collections import Counter
from pathlib import Path


def find_final_reports(base_dir, pattern=None):
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f'Base directory does not exist: {base_dir}')

    # Find all final_report.json files
    all_reports = list(base_path.rglob('final_report.json'))

    if pattern is None:
        return all_reports

    # Filter by pattern
    filtered_reports = []
    for report in all_reports:
        # Get relative path from base_dir for matching
        rel_path = report.relative_to(base_path)
        if fnmatch.fnmatch(str(rel_path), pattern):
            filtered_reports.append(report)

    return filtered_reports


def collect_resolved_ids(report_files):
    id_counter = Counter()

    for report_file in report_files:
        with open(report_file, 'r') as f:
            data = json.load(f)
            if 'resolved_ids' not in data:
                raise KeyError(f"'resolved_ids' key not found in {report_file}")
            resolved_ids = data['resolved_ids']
            id_counter.update(resolved_ids)

    return id_counter


def get_skip_ids(id_counter, threshold):
    return [id_str for id_str, count in id_counter.items() if count >= threshold]


def main():
    parser = argparse.ArgumentParser(
        description='Compute SKIP_IDS from resolved IDs in final_report.json files'
    )
    parser.add_argument(
        'threshold',
        type=int,
        help='Minimum number of times an ID must be resolved to be skipped',
    )
    parser.add_argument(
        '--base-dir',
        default='evaluation/evaluation_outputs/outputs',
        help='Base directory to search for final_report.json files (default: evaluation/evaluation_outputs/outputs)',
    )
    parser.add_argument(
        '--pattern',
        default=None,
        help='Glob pattern to filter paths (e.g., "*Multi-SWE-RL*/**/*gpt*")',
    )

    args = parser.parse_args()
    report_files = find_final_reports(args.base_dir, args.pattern)
    id_counter = collect_resolved_ids(report_files)

    skip_ids = get_skip_ids(id_counter, args.threshold)
    skip_ids = [s.replace('/', '__').replace(':pr-', '-') for s in skip_ids]
    skip_ids = ','.join(sorted(skip_ids))
    print(skip_ids)


if __name__ == '__main__':
    main()

import re
import argparse

from collector import FileCollector
from comparator import ComparisonExecutor


def parse_args():
    parser = argparse.ArgumentParser(description="Performance Metrics Comparator")

    parser.add_argument("dir_path_a", type=str, help="Directory containing analyzed results")
    parser.add_argument("dir_path_b", type=str, help="Directory containing analyzed results")
    parser.add_argument("--output", type=str, help="Output Directory after comparing")

    return parser.parse_args()


def main():
    args = parse_args()

    file_pattern = re.compile(r'(batch|service|request)_summary[.]csv|profiler[.]db')
    file_collector = FileCollector(pattern=file_pattern)

    executor = ComparisonExecutor(file_collector)
    executor.submit(args.dir_path_a, args.dir_path_b)


if __name__ == '__main__':
    main()

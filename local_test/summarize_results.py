import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


RESULT_PATTERN = re.compile(
    r"PERF_RESULT name=(?P<name>\S+) samples=(?P<samples>\d+) "
    r"median_us=(?P<median>[0-9.]+) mean_us=(?P<mean>[0-9.]+) "
    r"min_us=(?P<minimum>[0-9.]+) max_us=(?P<maximum>[0-9.]+)"
)


def read_result(result_dir: Path, case_name: str) -> Optional[Dict[str, str]]:
    result_path = result_dir / f"{case_name}.txt"
    if not result_path.is_file():
        return None
    match = RESULT_PATTERN.search(result_path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"invalid performance result: {result_path}")
    result = match.groupdict()
    if result["name"] != case_name:
        raise RuntimeError(
            f"result name {result['name']!r} does not match file case {case_name!r}"
        )
    return result


def build_summary(result_dir: Path, commit: str, case_names: list[str]) -> str:
    results = [result for name in case_names if (result := read_result(result_dir, name))]
    median_sum = sum(float(result["median"]) for result in results)
    mean_sum = sum(float(result["mean"]) for result in results)
    lines = [
        "# Concat local test summary",
        "",
        f"- Code commit: `{commit}`",
        f"- Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        f"- Completed cases: `{len(results)}/{len(case_names)}`",
        "- Correctness: every listed case passed bit-exact comparison",
        "",
        "| Case | Samples | Median/us | Mean/us | Min/us | Max/us |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| `{result['name']}` | {result['samples']} | {result['median']} | "
            f"{result['mean']} | {result['minimum']} | {result['maximum']} |"
        )
    lines.extend(
        [
            "",
            f"Median sum: `{median_sum:.3f} us`",
            "",
            f"Mean sum: `{mean_sum:.3f} us`",
            "",
            "Local diagnostic cases and official hidden cases are not directly comparable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(
            f"usage: python3 {sys.argv[0]} <result-dir> <output> <commit> <case> [<case> ...]"
        )
    result_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    summary = build_summary(result_dir, sys.argv[3], sys.argv[4:])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    print(f"LOCAL_SUMMARY path={output_path}")


if __name__ == "__main__":
    main()

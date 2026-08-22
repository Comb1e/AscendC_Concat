import csv
import statistics
import sys
from pathlib import Path


WARMUP_ROUNDS = 10
MEASURED_ROUNDS = 20


def read_concat_durations(profile_dir: Path) -> list[float]:
    durations = []
    for csv_path in profile_dir.rglob("op_summary*.csv"):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                op_name = row.get("Op Name", "")
                if "concat" not in op_name.lower():
                    continue
                duration = row.get("Task Duration(us)")
                if duration:
                    durations.append(float(duration))
    return durations


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: python3 {sys.argv[0]} <profile-dir> <case-name>")

    profile_dir = Path(sys.argv[1])
    case_name = sys.argv[2]
    durations = read_concat_durations(profile_dir)
    required = WARMUP_ROUNDS + MEASURED_ROUNDS
    if len(durations) < required:
        raise RuntimeError(
            f"found {len(durations)} Concat tasks under {profile_dir}, expected at least {required}; "
            "check the preceding application log for an ACLNN load or execution failure"
        )

    measured = durations[WARMUP_ROUNDS:required]
    print(
        f"PERF_RESULT name={case_name} samples={len(measured)} "
        f"median_us={statistics.median(measured):.3f} "
        f"mean_us={statistics.fmean(measured):.3f} "
        f"min_us={min(measured):.3f} max_us={max(measured):.3f}"
    )


if __name__ == "__main__":
    main()

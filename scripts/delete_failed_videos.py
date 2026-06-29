import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Delete failed/corrupted videos listed in a failed_videos.txt file."
    )
    parser.add_argument(
        "--failed-list",
        type=Path,
        default=Path("logs/landmark_extraction/failed_videos_latest.txt"),
        help="Text file containing one failed video path per line.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Without this flag, the script only previews what would be deleted.",
    )
    args = parser.parse_args()

    if not args.failed_list.exists():
        raise FileNotFoundError(f"Failed-list file not found: {args.failed_list}")

    paths = [
        Path(line.strip())
        for line in args.failed_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not paths:
        print(f"No failed videos listed in {args.failed_list}")
        return

    print(f"Found {len(paths)} failed video paths in {args.failed_list}")
    print()

    deleted_count = 0
    missing_count = 0

    for path in paths:
        if not path.exists():
            print(f"MISSING: {path}")
            missing_count += 1
            continue

        if args.delete:
            path.unlink()
            print(f"DELETED: {path}")
            deleted_count += 1
        else:
            print(f"WOULD DELETE: {path}")

    print()
    if args.delete:
        print(f"Deleted: {deleted_count}")
        print(f"Missing: {missing_count}")
    else:
        print("Dry run only. No files were deleted.")
        print("To actually delete these files, rerun with --delete.")


if __name__ == "__main__":
    main()

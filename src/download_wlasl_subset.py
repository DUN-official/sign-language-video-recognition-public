import argparse
import json
import logging
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def request_video(url: str, referer: str = ""):
    headers = {"User-Agent": "Mozilla/5.0"}
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, None, headers)
    response = urllib.request.urlopen(request, timeout=60)
    return response.read()


def existing_video(output_dir: Path, video_id: str):
    for ext in [".mp4", ".mkv", ".webm", ".swf"]:
        candidate = output_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def download_youtube(url: str, output_dir: Path, video_id: str):
    if existing_video(output_dir, video_id):
        logging.info("%s already exists", video_id)
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = output_dir / f"{video_id}.%(ext)s"
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        url,
        "-f",
        "mp4/best",
        "-o",
        str(output_template),
    ]
    return subprocess.run(cmd, check=False).returncode == 0


def download_direct(url: str, output_dir: Path, video_id: str):
    if existing_video(output_dir, video_id):
        logging.info("%s already exists", video_id)
        return True

    suffix = ".swf" if "aslpro" in url else ".mp4"
    path = output_dir / f"{video_id}{suffix}"
    referer = "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi" if "aslpro" in url else ""
    data = request_video(url, referer=referer)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def download_subset(index_path: Path, output_dir: Path, limit: Optional[int] = None):
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = []
    for entry in entries:
        for inst in entry["instances"]:
            instances.append((entry["gloss"], inst))

    if limit is not None:
        instances = instances[:limit]

    total = 0
    ok = 0
    failed = []

    for gloss, inst in instances:
        total += 1
        url = inst["url"]
        video_id = str(inst["video_id"])
        logging.info("Downloading gloss=%s video_id=%s", gloss, video_id)

        try:
            if "youtube" in url or "youtu.be" in url:
                success = download_youtube(url, output_dir, video_id)
            else:
                success = download_direct(url, output_dir, video_id)

            if success:
                ok += 1
            else:
                failed.append(video_id)
        except Exception as exc:
            logging.error("Failed video_id=%s error=%s", video_id, exc)
            failed.append(video_id)

        time.sleep(random.uniform(0.5, 1.5))

    missing_path = output_dir.parent / "missing_downloads.txt"
    missing_path.write_text("\n".join(failed), encoding="utf-8")

    print(f"Total attempted: {total}")
    print(f"Downloaded or already present: {ok}")
    print(f"Failed: {len(failed)}")
    print(f"Missing list: {missing_path}")


def main():
    parser = argparse.ArgumentParser(description="Download videos for a filtered WLASL subset.")
    parser.add_argument("--index", type=Path, required=True, help="Filtered WLASL subset JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional small test limit.")
    args = parser.parse_args()

    version_check = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if version_check.returncode != 0:
        raise RuntimeError(
            "yt-dlp is not available for this Python interpreter. "
            f"Install it with: {sys.executable} -m pip install yt-dlp"
        )

    download_subset(args.index, args.output_dir, args.limit)


if __name__ == "__main__":
    main()

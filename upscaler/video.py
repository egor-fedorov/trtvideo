"""Video metadata retrieval via ffprobe."""

import json
import subprocess
import sys


def get_video_info(input_path: str) -> dict:
    """Get video metadata via ffprobe.

    Args:
        input_path: Path to the video file.

    Returns:
        Dict with keys: fps, fps_str, nb_frames, width, height.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffprobe failed: {result.stderr}")
        sys.exit(1)

    info = json.loads(result.stdout)
    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            fps_str = stream.get("r_frame_rate", "30/1")
            num, den = map(int, fps_str.split("/"))
            fps = num / den

            nb_frames = int(stream.get("nb_frames", 0))
            width = int(stream["width"])
            height = int(stream["height"])

            return {
                "fps": fps,
                "fps_str": fps_str,
                "nb_frames": nb_frames,
                "width": width,
                "height": height,
            }

    print("ERROR: No video stream found")
    sys.exit(1)

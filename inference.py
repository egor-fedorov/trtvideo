#!/usr/bin/env python3
"""TensorRT video upscaler with ffmpeg pipe decode/encode.

Usage:
    upscale-video --engine models/engines/model_720p.engine --input video.mp4
"""

from upscaler.ffmpeg_pipeline import FfmpegPipeline


def main():
    FfmpegPipeline().run()


if __name__ == "__main__":
    main()

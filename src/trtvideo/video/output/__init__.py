"""Output-container preservation, muxing, and atomic publication."""

from trtvideo.video.output.preservation import (
    MediaPreservationError,
    build_container_preflight_command,
    build_ffmpeg_stream_copy_args,
    preflight_output_container,
)
from trtvideo.video.output.transaction import (
    AtomicOutputTransaction,
    commit_atomic_output,
    create_staging_output,
)

__all__ = [
    "AtomicOutputTransaction",
    "MediaPreservationError",
    "build_container_preflight_command",
    "build_ffmpeg_stream_copy_args",
    "commit_atomic_output",
    "create_staging_output",
    "preflight_output_container",
]

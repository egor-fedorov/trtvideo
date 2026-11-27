"""Reject unexpected release files and anything except our source and packaging metadata."""

import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


def verify(directory: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    package = f"trtvideo-{version}"
    wheel = directory / f"{package}-py3-none-any.whl"
    sdist = directory / f"{package}.tar.gz"
    if set(directory.iterdir()) != {wheel, sdist}:
        raise ValueError(f"expected only {wheel.name} and {sdist.name}")
    sources = {
        path.relative_to(root / "src").as_posix() for path in (root / "src/trtvideo").rglob("*.py")
    }
    with zipfile.ZipFile(wheel) as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        metadata = f"{package}.dist-info/"
        required = sources | {
            metadata + name
            for name in ("METADATA", "WHEEL", "RECORD", "licenses/LICENSE", "entry_points.txt")
        }
        unexpected = names - required - {metadata + "top_level.txt"}
        if unexpected or required - names:
            raise ValueError(
                f"invalid wheel contents: extra={unexpected}, missing={required - names}"
            )
    with tarfile.open(sdist) as archive:
        files = set()
        for info in archive:
            path = PurePosixPath(info.name).relative_to(package)
            if ".." in path.parts or not (info.isdir() or info.isfile()):
                raise ValueError(f"unsafe sdist entry: {info.name}")
            if not info.isdir():
                files.add(str(path))
        expected = {"src/" + file for file in sources}
        expected.update({"LICENSE", "README.md", "PKG-INFO", "pyproject.toml"})
        allowed = expected | {"setup.cfg"}
        allowed.update(
            "src/trtvideo.egg-info/" + name
            for name in (
                "PKG-INFO",
                "SOURCES.txt",
                "dependency_links.txt",
                "entry_points.txt",
                "requires.txt",
                "top_level.txt",
            )
        )
        if files - allowed or expected - files:
            raise ValueError(
                f"invalid sdist contents: extra={files - allowed}, missing={expected - files}"
            )


if __name__ == "__main__":
    verify(Path(sys.argv[1]))

"""Local promotion rehearsal: a real loopback registry, tiny indexes and a simulated GPU gate."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/release-images.sh"
REGISTRY = "registry:2@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373"
INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
VERSION = "1.2.3"
REVISION = "a" * 40


def descriptor(media_type, data):
    return {
        "mediaType": media_type,
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def request(url, method, data=None, *, media_type=INDEX):
    req = Request(
        url, data=data, method=method, headers={"Content-Type": media_type, "Accept": INDEX}
    )
    with urlopen(req, timeout=10) as response:
        return response.headers


def seed(repo, tag, generation):
    """Upload a scratch linux/amd64 index; registry behavior is not mocked."""
    api = f"http://{repo.split('/', 1)[0]}/v2/{repo.split('/', 1)[1]}"
    platform = {"os": "linux", "architecture": "amd64"}
    config = json.dumps(
        {
            **platform,
            "rootfs": {"type": "layers", "diff_ids": []},
            "config": {"Labels": {"generation": generation, "repo": repo}},
        }
    ).encode()
    blob = descriptor(CONFIG, config)
    upload = request(f"{api}/blobs/uploads/", "POST")["Location"]
    separator = "&" if "?" in upload else "?"
    request(
        upload + separator + "digest=" + blob["digest"],
        "PUT",
        config,
        media_type="application/octet-stream",
    )
    manifest = json.dumps(
        {"schemaVersion": 2, "mediaType": MANIFEST, "config": blob, "layers": []}
    ).encode()
    image = descriptor(MANIFEST, manifest)
    request(f"{api}/manifests/{image['digest']}", "PUT", manifest, media_type=MANIFEST)
    index = json.dumps(
        {"schemaVersion": 2, "mediaType": INDEX, "manifests": [{**image, "platform": platform}]}
    ).encode()
    return request(f"{api}/manifests/{tag}", "PUT", index)["Docker-Content-Digest"]


def read_tag(repo, tag):
    host, path = repo.split("/", 1)
    try:
        return request(f"http://{host}/v2/{path}/manifests/{tag}", "HEAD")["Docker-Content-Digest"]
    except HTTPError as exc:
        if exc.code != 404:
            raise
        exc.close()
        return None


class PromotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        container = subprocess.check_output(
            ["docker", "run", "--rm", "-d", "-p", "127.0.0.1::5000", REGISTRY], text=True
        ).strip()
        cls.addClassCleanup(subprocess.run, ["docker", "stop", container], check=True)
        cls.host = subprocess.check_output(
            ["docker", "port", container, "5000/tcp"], text=True
        ).strip()
        assert cls.host.startswith("127.0.0.1:")
        for _ in range(50):
            try:
                request(f"http://{cls.host}/v2/", "GET")
                break
            except URLError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Loopback registry did not become ready")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repos = [f"{self.host}/{self._testMethodName}/{name}" for name in ("prod", "tools")]
        self.old = [seed(repo, "latest", "old") for repo in self.repos]
        self.new = [seed(repo, "candidate-123", "new") for repo in self.repos]
        self.refs = [f"{r}@{d}" for r, d in zip(self.repos, self.new, strict=True)]
        (self.root / "Makefile").write_text(
            f'test-release-gpu:\n\t@test "$(IMAGE)" = {self.refs[0]}\n'
            f'\t@test "$(MODEL_TOOLS_IMAGE)" = {self.refs[1]}\n\t@test -z "$$FAIL_TOOLS"\n'
        )
        self.env = {
            **os.environ,
            "PATH": f"{self.root}:{os.environ['PATH']}",
            "REAL_DOCKER": shutil.which("docker"),
            "GITHUB_REPOSITORY": "local/rehearsal",
            "CURRENT_RELEASE": f"v{VERSION}",
            "FAIL_TOOLS": "",
        }
        gh = self.root / "gh"
        gh.write_text(
            '#!/bin/sh\ntest "$*" = "api repos/local/rehearsal/releases/latest --jq .tag_name" '
            '|| exit 2\necho "$CURRENT_RELEASE"\n'
        )
        gh.chmod(0o700)

    def publish(self, *, prerelease=False, fails=False, **env):
        result = subprocess.run(
            ["bash", SCRIPT, "publish", *self.refs, VERSION, REVISION, str(prerelease).lower()],
            cwd=self.root,
            env={**self.env, **env},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode == 0, not fails, result.stdout + result.stderr)

    def tags(self, tag):
        return [read_tag(r, tag) for r in self.repos]

    def test_record_candidate(self):
        command = ["bash", SCRIPT, "candidate", self.refs[0], "candidate-456"]
        for _ in range(2):
            subprocess.run(command, check=True, capture_output=True)
        command[3] = f"{self.repos[0]}@{self.old[0]}"
        conflict = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(conflict.returncode, 0)
        self.assertIn("Refusing to replace", conflict.stderr)
        self.assertEqual(self.tags("candidate-456"), [self.new[0], None])
        self.assertEqual(self.tags("latest"), self.old)

    def test_stable_and_retry(self):
        for _ in range(2):
            self.publish()
            for tag in (VERSION, f"v{VERSION}", f"sha-{REVISION}", "latest"):
                self.assertEqual(self.tags(tag), self.new)

    def test_prerelease_and_old_stable(self):
        self.publish(prerelease=True)
        self.assertEqual(self.tags("latest"), self.old)
        self.assertEqual(self.tags(f"v{VERSION}"), self.new)
        self.publish(CURRENT_RELEASE="v2.0.0")
        self.assertEqual(self.tags("latest"), self.old)

    def test_second_gate_failure(self):
        self.publish(fails=True, FAIL_TOOLS="1")
        self.assertEqual(self.tags("latest"), self.old)
        for tag in (VERSION, f"v{VERSION}", f"sha-{REVISION}"):
            self.assertEqual(self.tags(tag), [None, None])

    def test_conflict_in_second_package(self):
        conflict = seed(self.repos[1], f"v{VERSION}", "different")
        self.publish(fails=True)
        self.assertEqual(self.tags("latest"), self.old)
        self.assertEqual(self.tags(VERSION), [None, None])
        self.assertEqual(self.tags(f"v{VERSION}"), [None, conflict])

    def test_interrupted_latest(self):
        docker = self.root / "docker"
        docker.write_text(
            '#!/bin/sh\nif [ "$3" = create ] && [ "$6" = "$FAIL_WRITE" ]; then exit 75; fi\n'
            'exec "$REAL_DOCKER" "$@"\n'
        )
        docker.chmod(0o700)
        self.publish(fails=True, FAIL_WRITE=self.repos[1] + ":latest")
        self.assertEqual(self.tags("latest"), [self.new[0], self.old[1]])
        self.publish()
        self.assertEqual(self.tags("latest"), self.new)


if __name__ == "__main__":
    unittest.main()

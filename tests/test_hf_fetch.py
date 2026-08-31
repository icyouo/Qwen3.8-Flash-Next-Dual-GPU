from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from q38_hf_fetch import fetch_one, git_blob_sha1  # noqa: E402


class HuggingFaceFetchTest(unittest.TestCase):
    def test_non_lfs_file_is_bound_by_git_blob_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            path.write_bytes(b'{"model":"q38"}\n')
            digest = hashlib.sha1(
                f"blob {path.stat().st_size}\0".encode() + path.read_bytes()
            ).hexdigest()
            self.assertEqual(git_blob_sha1(path), digest)
            name, status = fetch_one(
                "unused/repo",
                "0" * 40,
                root,
                {
                    "rfilename": path.name,
                    "size": path.stat().st_size,
                    "blobId": digest,
                },
            )
            self.assertEqual((name, status), (path.name, "present"))


if __name__ == "__main__":
    unittest.main()

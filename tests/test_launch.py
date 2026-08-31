from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from q38_launch import (  # noqa: E402
    PINNED_COMMIT,
    PINNED_REPO,
    runtime_command,
    sha256_file,
    validate_ready,
)
from q38_identity import (  # noqa: E402
    fnv64,
    hash_group,
    hash_path,
    hash_runtime,
    sampling_config,
)


class LaunchManifestTest(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        segment0 = root / "stage0" / "segments" / "a.bin"
        segment1 = root / "stage1" / "segments" / "b.bin"
        segment0.parent.mkdir(parents=True)
        segment1.parent.mkdir(parents=True)
        segment0.write_bytes(b"stage0")
        segment1.write_bytes(b"stage1")
        index0 = root / "stage0" / "index.q38d"
        index1 = root / "stage1" / "index.q38d"
        index0.write_text(
            "Q38_DEVICE_INDEX_V1\n"
            f"segment\tsegments/a.bin\t6\t{sha256_file(segment0)}\n"
        )
        index1.write_text(
            "Q38_DEVICE_INDEX_V1\n"
            f"segment\tsegments/b.bin\t6\t{sha256_file(segment1)}\n"
        )
        ple = root / "ple.q38p"
        ple.write_text("ple")
        metadata = root / "metadata"
        metadata.mkdir()
        (metadata / "config.json").write_text("{}")
        model_root = root / "source"
        model_root.mkdir()
        tokenizer = model_root / "tokenizer.json"
        tokenizer_config = model_root / "tokenizer_config.json"
        tokenizer.write_text("{}")
        tokenizer_config.write_text("{}")
        identity = root / "session.q38i"
        sampling = sampling_config('{"mode":"greedy"}')
        digest_bytes = {
            "model_checkpoint": hash_path(metadata),
            "tokenizer": hash_group([tokenizer, tokenizer_config]),
            "chat_template": hash_path(tokenizer_config),
            "runtime": hash_runtime(TOOLS.parent),
            "kernels": hash_path(TOOLS.parent / "cuda"),
            "ple_layout": hash_path(ple),
            "stage_plan": hash_group([index0, index1]),
            "sampling_parser": hashlib.sha256(
                json.dumps(sampling, sort_keys=True, separators=(",", ":")).encode()
            ).digest(),
        }
        raw = (
            struct.pack("<IHHQ", int.from_bytes(b"Q38I", "little"), 1, 288, 99)
            + b"".join(digest_bytes[name] for name in (
                "model_checkpoint", "tokenizer", "chat_template", "runtime",
                "kernels", "ple_layout", "stage_plan", "sampling_parser"
            ))
            + struct.pack("<IIQ", 262144, 0, 0)
        )
        identity.write_text(
            "Q38_SESSION_IDENTITY_V1\n"
            "session_hash=99\n"
            + "".join(f"{name}_sha256={value.hex()}\n" for name, value in digest_bytes.items())
            + f"context_limit=262144\nflags=0\nidentity_checksum={fnv64(raw)}\n"
        )
        ready = {
            "schema": "Q38_PRODUCTION_ARTIFACT_READY_V1",
            "model": "Qwen3.8-Flash-Next",
            "source_repo": PINNED_REPO,
            "source_commit": PINNED_COMMIT,
            "cut": 25,
            "context_limit": 262144,
            "vocabulary": 248320,
            "sampling": sampling,
            "source_model_root": str(model_root),
            "model_metadata": str(metadata),
            "tokenizer_files": [str(tokenizer), str(tokenizer_config)],
            "chat_template": str(tokenizer_config),
            "stage0_index": str(index0),
            "stage1_index": str(index1),
            "ple_layout": str(ple),
            "session_identity": str(identity),
            "stage0_index_sha256": sha256_file(index0),
            "stage1_index_sha256": sha256_file(index1),
            "ple_layout_sha256": sha256_file(ple),
            "identity_sha256": sha256_file(identity),
        }
        path = root / "READY.json"
        path.write_text(json.dumps(ready))
        return path

    def test_validates_all_ready_digests_and_builds_bound_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.fixture(Path(temporary))
            validated = validate_ready(path)
            self.assertEqual(validated["session_hash"], 99)
            self.assertEqual(validated["segments"], 2)
            command = runtime_command(
                validated,
                Path("/bin/true"),
                Path("/tmp/q38.sock"),
                Path("/tmp/q38.journal"),
                4096,
                8,
            )
            self.assertIn("--identity", command)
            self.assertIn("99", command)
            self.assertIn("--sampling", command)
            self.assertIn("--snapshot-journal", command)
            self.assertNotIn("--enable-mtp", command)
            fast_command = runtime_command(
                validated,
                Path("/bin/true"),
                Path("/tmp/q38.sock"),
                None,
                4096,
                8,
            )
            self.assertNotIn("--snapshot-journal", fast_command)
            mtp_command = runtime_command(
                validated,
                Path("/bin/true"),
                Path("/tmp/q38.sock"),
                Path("/tmp/q38.journal"),
                4096,
                8,
                True,
            )
            self.assertIn("--enable-mtp", mtp_command)

    def test_detects_segment_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.fixture(root)
            (root / "stage0" / "segments" / "a.bin").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                validate_ready(path)

    def test_rejects_partial_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.fixture(root)
            (root / "stage0" / "orphan.part").write_text("x")
            with self.assertRaises(ValueError):
                validate_ready(path)

    def test_runtime_identity_ignores_docs_tests_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Makefile").write_text("runtime build\n")
            for directory in ("include", "src", "cuda", "tools"):
                path = root / directory
                path.mkdir()
                (path / "input.txt").write_text(directory)
            initial = hash_runtime(root)
            (root / "README.md").write_text("documentation\n")
            (root / "READINESS.md").write_text("evidence notes\n")
            (root / "tests").mkdir()
            (root / "tests" / "test.txt").write_text("test\n")
            (root / "out").mkdir()
            (root / "out" / "hardware.log").write_text("mutable evidence\n")
            self.assertEqual(hash_runtime(root), initial)
            (root / "src" / "input.txt").write_text("changed production source\n")
            self.assertNotEqual(hash_runtime(root), initial)


if __name__ == "__main__":
    unittest.main()

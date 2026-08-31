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

from q38_prepare_artifact import (  # noqa: E402
    PINNED_COMMIT,
    PINNED_REPO,
    existing_fragment_complete,
    expected_fragment_stages,
    required_shards,
    stream_compile_shards,
    validate_fetch_manifest,
)
from q38_quant_policy import policy_identity  # noqa: E402


class PrepareArtifactTest(unittest.TestCase):
    def source(self, root: Path, names: list[str]) -> Path:
        header = {}
        payload = bytearray()
        for name in names:
            begin = len(payload)
            payload.extend(b"\0\0")
            header[name] = {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [begin, len(payload)],
            }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        path = root / "model-00001-of-00001.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return path

    def fragment(self, output: Path, source: Path, stage: str) -> tuple[Path, Path]:
        root = output / stage / "segments"
        root.mkdir(parents=True, exist_ok=True)
        stem = source.name.removesuffix(".safetensors")
        segment = root / f"{stem}.q38w"
        segment.write_bytes(stage.encode())
        fragment = root / f"{stem}.q38.json"
        fragment.write_text(
            json.dumps(
                {
                    "schema": "Q38_STAGE_FRAGMENT_V1",
                    "source_repo": PINNED_REPO,
                    "source_commit": PINNED_COMMIT,
                    "source_shard": source.name,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "policy_sha256": policy_identity(25),
                    "stage": int(stage[-1]),
                    "cut": 25,
                    "segment": segment.name,
                    "segment_bytes": segment.stat().st_size,
                    "segment_sha256": hashlib.sha256(segment.read_bytes()).hexdigest(),
                    "tensors": [],
                }
            )
        )
        return fragment, segment

    def test_dual_owner_shard_requires_both_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(
                root,
                [
                    "model.language_model.embed_tokens.weight",
                    "model.language_model.hyper_connection_mixer.hc_norm.weight",
                ],
            )
            output = root / "artifact"
            self.assertEqual(expected_fragment_stages(source, 25), {"stage0", "stage1"})
            self.fragment(output, source, "stage0")
            self.assertFalse(existing_fragment_complete(output, source, 25))
            self.fragment(output, source, "stage1")
            self.assertTrue(existing_fragment_complete(output, source, 25))

    def test_single_owner_fragment_is_reusable_but_corruption_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(
                root, ["model.language_model.hyper_connection_mixer.hc_norm.weight"]
            )
            output = root / "artifact"
            self.assertEqual(expected_fragment_stages(source, 25), {"stage1"})
            _fragment, segment = self.fragment(output, source, "stage1")
            self.assertTrue(existing_fragment_complete(output, source, 25))
            segment.write_bytes(b"changed")
            self.assertFalse(existing_fragment_complete(output, source, 25))

    def test_source_download_must_have_every_shard_and_no_partials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root, ["model.language_model.embed_tokens.weight"])
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": source.name}})
            )
            self.assertEqual(required_shards(root), [source])
            (root / "download.part").write_text("partial")
            with self.assertRaises(RuntimeError):
                required_shards(root)

    def test_malformed_or_escaping_fragment_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(
                root, ["model.language_model.hyper_connection_mixer.hc_norm.weight"]
            )
            output = root / "artifact"
            fragment, _segment = self.fragment(output, source, "stage1")
            fragment.write_text("not-json")
            self.assertFalse(existing_fragment_complete(output, source, 25))
            fragment.write_text(json.dumps({"segment": "../outside.q38w"}))
            self.assertFalse(existing_fragment_complete(output, source, 25))

    def test_fetch_manifest_must_bind_the_pinned_commit_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root, ["model.language_model.embed_tokens.weight"])
            manifest = {
                "schema": "Q38_HF_FETCH_V1",
                "repo": PINNED_REPO,
                "commit": PINNED_COMMIT,
                "files": [
                    {
                        "path": source.name,
                        "size": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            }
            (root / "q38-fetch-manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(validate_fetch_manifest(root, [source]), manifest)
            manifest["commit"] = "0" * 40
            (root / "q38-fetch-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                validate_fetch_manifest(root, [source])

    def test_streaming_compile_can_resume_after_verified_source_prune(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(
                root, ["model.language_model.hyper_connection_mixer.hc_norm.weight"]
            )
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.language_model.hyper_connection_mixer.hc_norm.weight":
                                source.name
                        }
                    }
                )
            )
            records = []
            for path in (source, index):
                records.append(
                    {
                        "path": path.name,
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            (root / "q38-fetch-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "Q38_HF_FETCH_V1",
                        "repo": PINNED_REPO,
                        "commit": PINNED_COMMIT,
                        "files": records,
                    }
                )
            )
            output = root / "artifact"
            self.assertEqual(
                stream_compile_shards(root, output, 25, False, True), 1
            )
            self.assertFalse(source.exists())
            self.assertEqual(
                stream_compile_shards(root, output, 25, False, True), 1
            )


if __name__ == "__main__":
    unittest.main()

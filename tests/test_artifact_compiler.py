import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import json
    from pathlib import Path
    import struct
    import sys
    import tempfile
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import q38_artifact_compiler as compiler


@unittest.skipIf(np is None, "NumPy is not installed")
class ArtifactCompilerTest(unittest.TestCase):
    def write_bf16_safetensor(self, root, name, values):
        words = compiler.f32_to_bf16_rne(np.asarray(values, dtype=np.float32))
        payload = words.tobytes(order="C")
        header = {
            name: {
                "dtype": "BF16",
                "shape": list(words.shape),
                "data_offsets": [0, len(payload)],
            }
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        path = root / "source.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return path

    def compile_fixture(self, source, output):
        return compiler.compile_source_shard(
            SimpleNamespace(
                source=source,
                source_sha256=None,
                source_commit="c" * 40,
                output=output,
                cut=25,
                plan_only=False,
            )
        )

    def test_bf16_roundtrip_is_bounded(self):
        source = np.array([-448.0, -1.5, -0.0, 0.125, 1.0, 448.0], dtype=np.float32)
        encoded = compiler.f32_to_bf16_rne(source)
        decoded = compiler.bf16_to_f32(encoded)
        np.testing.assert_allclose(decoded, source, rtol=0.008, atol=0.001)

    def test_fp8_known_encodings(self):
        source = np.array(
            [0.0, -0.0, 2.0**-9, 0.5, 1.0, 2.0, 448.0, -448.0],
            dtype=np.float32,
        )
        encoded = compiler.f32_to_e4m3fn(source)
        np.testing.assert_array_equal(
            encoded,
            np.array([0x00, 0x80, 0x01, 0x30, 0x38, 0x40, 0x7E, 0xFE], dtype=np.uint8),
        )

    def test_int4_packing_order(self):
        q = np.array([[-7, -1, 0, 7]], dtype=np.int8)
        low = q[:, 0::2].astype(np.int16) & 0xF
        high = q[:, 1::2].astype(np.int16) & 0xF
        packed = (low | (high << 4)).astype(np.uint8)
        np.testing.assert_array_equal(packed, np.array([[0xF9, 0x70]], dtype=np.uint8))

    def test_fp8_row_scale_preserves_small_ple_values(self):
        values = np.array(
            [
                [-0.044, -0.0075, -0.001, 0.0, 0.002, 0.009, 0.031, 0.043],
                [-0.0004, -0.0001, 0.0, 0.00008, 0.0002, 0.0003, 0.00035, 0.0004],
            ],
            dtype=np.float32,
        )
        encoded, scale_words = compiler.quantize_fp8_rows(values)
        scales = compiler.bf16_to_f32(scale_words)
        restored = compiler.e4m3fn_to_f32(encoded) * scales[:, None]
        np.testing.assert_allclose(restored, values, rtol=0.08, atol=2.0e-5)
        self.assertLess(float(scales[0]), 1.0e-3)

    def test_embedding_is_materialized_in_both_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.linspace(-0.25, 0.25, 2 * 128, dtype=np.float32).reshape(
                2, 128
            )
            source = self.write_bf16_safetensor(
                root, "model.language_model.embed_tokens.weight", values
            )
            fragments = self.compile_fixture(source, root / "artifact")
            self.assertEqual(len(fragments), 2)
            records = []
            for fragment_path in fragments:
                fragment = json.loads(fragment_path.read_text())
                self.assertEqual(len(fragment["tensors"]), 1)
                tensor = fragment["tensors"][0]
                self.assertEqual(tensor["name"], "token_embd.weight")
                self.assertEqual(tensor["format"], "w8a16_sym_g128")
                records.append(tensor)
            self.assertEqual(records[0]["data_sha256"], records[1]["data_sha256"])
            self.assertEqual(
                records[0]["scale_sha256"], records[1]["scale_sha256"]
            )

    def test_final_hyper_mixer_is_owned_by_stage1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.linspace(-0.1, 0.1, 16, dtype=np.float32)
            source = self.write_bf16_safetensor(
                root,
                "model.language_model.hyper_connection_mixer.hc_norm.weight",
                values,
            )
            fragments = self.compile_fixture(source, root / "artifact")
            self.assertEqual(len(fragments), 1)
            self.assertIn("stage1", fragments[0].parts)
            fragment = json.loads(fragments[0].read_text())
            self.assertEqual(fragment["stage"], 1)
            self.assertEqual(
                fragment["tensors"][0]["name"], "hc_input.norm.weight"
            )

    def test_ple_artifact_publishes_row_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.linspace(-0.044, 0.044, 2 * 160, dtype=np.float32).reshape(
                2, 160
            )
            source = self.write_bf16_safetensor(
                root,
                "model.language_model.layers.1.ple.ple_embedding."
                "ngram_embedding.shard_0.weight",
                values,
            )
            fragments = self.compile_fixture(source, root / "artifact")
            self.assertEqual(len(fragments), 1)
            fragment = json.loads(fragments[0].read_text())
            tensor = fragment["tensors"][0]
            self.assertEqual(tensor["format"], "fp8_e4m3fn")
            self.assertEqual(tensor["data_bytes"], 2 * 160)
            self.assertEqual(tensor["scale_bytes"], 2 * 2)
            segment = fragments[0].parent / fragment["segment"]
            with segment.open("rb") as source_file:
                source_file.seek(tensor["data_offset"])
                encoded = np.frombuffer(
                    source_file.read(tensor["data_bytes"]), dtype=np.uint8
                ).reshape(2, 160)
                source_file.seek(tensor["scale_offset"])
                scales = np.frombuffer(
                    source_file.read(tensor["scale_bytes"]), dtype=np.uint16
                )
            restored = compiler.e4m3fn_to_f32(encoded) * compiler.bf16_to_f32(
                scales
            )[:, None]
            np.testing.assert_allclose(restored, values, rtol=0.08, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()

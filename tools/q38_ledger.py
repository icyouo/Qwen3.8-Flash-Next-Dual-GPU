#!/usr/bin/env python3
"""Read split GGUF headers and build a Qwen3.8 stage-cut ledger.

The tool never maps or reads tensor payloads. It parses metadata and tensor
directories, verifies each declared extent against its shard, and accounts
bytes by layer/owner/type for candidate contiguous cuts.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterable


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

VALUE_UINT8 = 0
VALUE_INT8 = 1
VALUE_UINT16 = 2
VALUE_INT16 = 3
VALUE_UINT32 = 4
VALUE_INT32 = 5
VALUE_FLOAT32 = 6
VALUE_BOOL = 7
VALUE_STRING = 8
VALUE_ARRAY = 9
VALUE_UINT64 = 10
VALUE_INT64 = 11
VALUE_FLOAT64 = 12

# name, elements per block, bytes per block. The current Qwen artifact uses
# F32, Q5_0, Q8_0, Q4_K, Q5_K, I64 and BF16; common neighboring types are
# included so the ledger can compare future compiler outputs.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
}

SCALAR_FORMATS: dict[int, str] = {
    VALUE_UINT8: "B",
    VALUE_INT8: "b",
    VALUE_UINT16: "H",
    VALUE_INT16: "h",
    VALUE_UINT32: "I",
    VALUE_INT32: "i",
    VALUE_FLOAT32: "f",
    VALUE_BOOL: "B",
    VALUE_UINT64: "Q",
    VALUE_INT64: "q",
    VALUE_FLOAT64: "d",
}

KEEP_METADATA = {
    "general.architecture",
    "general.alignment",
    "split.count",
    "split.no",
    "split.tensors.count",
}


class GGUFError(RuntimeError):
    pass


class Reader:
    def __init__(self, file: BinaryIO, path: Path):
        self.file = file
        self.path = path

    def tell(self) -> int:
        return self.file.tell()

    def read_exact(self, count: int) -> bytes:
        data = self.file.read(count)
        if len(data) != count:
            raise GGUFError(f"{self.path}: truncated at byte {self.tell()}")
        return data

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read_exact(size))

    def u32(self) -> int:
        return int(self.unpack("I")[0])

    def u64(self) -> int:
        return int(self.unpack("Q")[0])

    def string(self) -> str:
        length = self.u64()
        if length > 1 << 30:
            raise GGUFError(f"{self.path}: unreasonable string length {length}")
        try:
            return self.read_exact(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise GGUFError(f"{self.path}: invalid UTF-8 string") from error

    def value(self, value_type: int, keep: bool) -> Any:
        if value_type in SCALAR_FORMATS:
            fmt = SCALAR_FORMATS[value_type]
            value = self.unpack(fmt)[0]
            return value if keep else None
        if value_type == VALUE_STRING:
            value = self.string()
            return value if keep else None
        if value_type == VALUE_ARRAY:
            element_type = self.u32()
            length = self.u64()
            if length > 100_000_000:
                raise GGUFError(f"{self.path}: unreasonable array length {length}")
            retained = keep and length <= 4096
            result = [] if retained else None
            for _ in range(length):
                value = self.value(element_type, retained)
                if retained:
                    result.append(value)
            return result
        raise GGUFError(f"{self.path}: unknown metadata type {value_type}")


@dataclasses.dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    type_id: int
    type_name: str
    offset: int
    payload_bytes: int
    shard: str
    data_start: int

    @property
    def elements(self) -> int:
        return math.prod(self.dims)


@dataclasses.dataclass
class ShardInfo:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    data_start: int
    file_bytes: int


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise GGUFError(f"invalid GGUF alignment {alignment}")
    return (value + alignment - 1) & ~(alignment - 1)


def tensor_bytes(dims: Iterable[int], type_id: int) -> int:
    if type_id not in GGML_TYPES:
        raise GGUFError(f"unsupported GGML tensor type {type_id}")
    _, block_elements, block_bytes = GGML_TYPES[type_id]
    elements = math.prod(dims)
    return (elements + block_elements - 1) // block_elements * block_bytes


def parse_shard(path: Path) -> ShardInfo:
    file_bytes = path.stat().st_size
    with path.open("rb") as file:
        reader = Reader(file, path)
        if reader.read_exact(4) != GGUF_MAGIC:
            raise GGUFError(f"{path}: not a GGUF file")
        version = reader.u32()
        if version != GGUF_VERSION:
            raise GGUFError(f"{path}: only GGUF v3 is supported, got {version}")
        tensor_count = reader.u64()
        metadata_count = reader.u64()
        if tensor_count > 10_000_000 or metadata_count > 10_000_000:
            raise GGUFError(f"{path}: unreasonable header counts")

        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.string()
            value_type = reader.u32()
            keep = key in KEEP_METADATA or key.startswith("qwen4exp.")
            value = reader.value(value_type, keep)
            if keep:
                metadata[key] = value

        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = reader.string()
            dimensions = reader.u32()
            if dimensions == 0 or dimensions > 8:
                raise GGUFError(f"{path}: tensor {name} has {dimensions} dimensions")
            dims = tuple(reader.u64() for _ in range(dimensions))
            type_id = reader.u32()
            offset = reader.u64()
            raw_tensors.append((name, dims, type_id, offset))

        alignment = int(metadata.get("general.alignment", 32))
        data_start = align_up(reader.tell(), alignment)
        tensors: list[TensorInfo] = []
        for name, dims, type_id, offset in raw_tensors:
            payload_bytes = tensor_bytes(dims, type_id)
            end = data_start + offset + payload_bytes
            if end > file_bytes:
                raise GGUFError(
                    f"{path}: tensor {name} ends at {end}, beyond {file_bytes}"
                )
            tensors.append(
                TensorInfo(
                    name=name,
                    dims=dims,
                    type_id=type_id,
                    type_name=GGML_TYPES[type_id][0],
                    offset=offset,
                    payload_bytes=payload_bytes,
                    shard=path.name,
                    data_start=data_start,
                )
            )
        return ShardInfo(path, version, metadata, tensors, data_start, file_bytes)


SPLIT_RE = re.compile(r"^(.*-)(\d{5})(-of-)(\d{5})(\.gguf)$")


def discover_shards(model: Path) -> list[Path]:
    if model.is_dir():
        shards = sorted(model.glob("*.gguf"))
        if not shards:
            raise GGUFError(f"{model}: directory contains no GGUF files")
        return shards
    match = SPLIT_RE.match(model.name)
    if not match:
        if not model.is_file():
            raise GGUFError(f"{model}: model file does not exist")
        return [model]
    total = int(match.group(4))
    paths = [
        model.with_name(
            f"{match.group(1)}{index:05d}{match.group(3)}{total:05d}{match.group(5)}"
        )
        for index in range(1, total + 1)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise GGUFError("missing split GGUF shards: " + ", ".join(missing))
    return paths


LAYER_RE = re.compile(r"(?:^|\.)(?:blk|layers)\.(\d+)(?:\.|$)")


def classify_tensor(name: str) -> tuple[str, int | None]:
    lower = name.lower()
    if lower.startswith(("v.", "vblk.", "vision.", "visual.")) or ".vision." in lower:
        return "vision", None
    if lower.startswith("mtp.") or ".mtp." in lower or "nextn" in lower:
        return "mtp", None
    layer = LAYER_RE.search(lower)
    if layer:
        return "layer", int(layer.group(1))
    if (lower.startswith("token_embd") or "input_hc" in lower or
            lower.startswith("hc_input") or lower.startswith("position")):
        return "stage0_global", None
    if lower.startswith("output") or "lm_head" in lower or "final_norm" in lower:
        return "stage1_global", None
    return "unassigned", None


def gib(value: int) -> float:
    return value / (1 << 30)


def build_ledger(
    shards: list[ShardInfo], cuts: list[int], context_limit: int,
    artifact_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tensors = [tensor for shard in shards for tensor in shard.tensors]
    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        raise GGUFError("duplicate tensor names across shards: " + ", ".join(duplicates[:8]))

    bytes_by_type: Counter[str] = Counter()
    tensors_by_type: Counter[str] = Counter()
    bytes_by_class: Counter[str] = Counter()
    bytes_by_layer: defaultdict[int, int] = defaultdict(int)
    class_rows: list[dict[str, Any]] = []
    for tensor in tensors:
        owner, layer = classify_tensor(tensor.name)
        bytes_by_type[tensor.type_name] += tensor.payload_bytes
        tensors_by_type[tensor.type_name] += 1
        bytes_by_class[owner] += tensor.payload_bytes
        if layer is not None:
            bytes_by_layer[layer] += tensor.payload_bytes
        class_rows.append(
            {
                "name": tensor.name,
                "dims": list(tensor.dims),
                "type": tensor.type_name,
                "payload_bytes": tensor.payload_bytes,
                "owner_class": owner,
                "layer": layer,
                "shard": tensor.shard,
                "offset": tensor.offset,
            }
        )

    metadata: dict[str, Any] = {}
    for shard in shards:
        for key, value in shard.metadata.items():
            if key not in metadata:
                metadata[key] = value
            elif metadata[key] != value and not key.startswith("split."):
                raise GGUFError(f"metadata {key} differs between shards")

    layer_count = int(metadata.get("qwen4exp.block_count", 48))
    qsa_layers = metadata.get("qwen4exp.attention.full_layers")
    if not isinstance(qsa_layers, list):
        qsa_layers = list(range(3, layer_count, 4))
    qsa_layers = [int(layer) for layer in qsa_layers]
    layer_set = set(bytes_by_layer)
    missing_layers = [layer for layer in range(layer_count) if layer not in layer_set]

    stage0_global = bytes_by_class["stage0_global"]
    stage1_global = bytes_by_class["stage1_global"] + bytes_by_class["mtp"]
    unassigned = bytes_by_class["unassigned"]
    plans = []
    for cut in cuts:
        if cut <= 0 or cut >= layer_count:
            raise GGUFError(f"cut {cut} is outside 1..{layer_count - 1}")
        stage0_weights = stage0_global + sum(
            size for layer, size in bytes_by_layer.items() if layer < cut
        )
        stage1_weights = stage1_global + sum(
            size for layer, size in bytes_by_layer.items() if layer >= cut
        )
        qsa0 = sum(layer < cut for layer in qsa_layers)
        qsa1 = len(qsa_layers) - qsa0
        # Main BF16 K/V is 2048 B/token/layer; compressed BF16 index is
        # 64 B/token/layer. Raw rings and metadata are bounded and reported
        # separately by the runtime planner.
        qsa0_state = qsa0 * context_limit * (2048 + 64)
        qsa1_state = qsa1 * context_limit * (2048 + 64)
        plans.append(
            {
                "cut": cut,
                "stage0_layers": [0, cut - 1],
                "stage1_layers": [cut, layer_count - 1],
                "stage0_weight_bytes": stage0_weights,
                "stage1_weight_bytes": stage1_weights,
                "stage0_qsa_layers": qsa0,
                "stage1_qsa_layers": qsa1,
                "stage0_qsa_state_bytes": qsa0_state,
                "stage1_qsa_state_bytes": qsa1_state,
                "stage0_weight_plus_qsa_bytes": stage0_weights + qsa0_state,
                "stage1_weight_plus_qsa_bytes": stage1_weights + qsa1_state,
                "weight_imbalance_bytes": abs(stage0_weights - stage1_weights),
            }
        )

    layout_hash = hashlib.sha256()
    for tensor in sorted(tensors, key=lambda item: item.name):
        layout_hash.update(tensor.name.encode())
        layout_hash.update(b"\0")
        layout_hash.update(str(tensor.dims).encode())
        layout_hash.update(struct.pack("<IQ", tensor.type_id, tensor.payload_bytes))

    ple = None
    if artifact_manifest:
        ple = artifact_manifest.get("ple_sidecar")

    return {
        "schema_version": 1,
        "model": {
            "architecture": metadata.get("general.architecture"),
            "layer_count": layer_count,
            "context_limit": context_limit,
            "qsa_layers": qsa_layers,
            "shard_count": len(shards),
            "tensor_count": len(tensors),
            "tensor_payload_bytes": sum(t.payload_bytes for t in tensors),
            "file_bytes": sum(shard.file_bytes for shard in shards),
            "layout_sha256": layout_hash.hexdigest(),
            "missing_layer_directories": missing_layers,
        },
        "ple_sidecar": ple,
        "by_type": {
            name: {"tensors": tensors_by_type[name], "bytes": bytes_by_type[name]}
            for name in sorted(bytes_by_type)
        },
        "by_owner_class": dict(sorted(bytes_by_class.items())),
        "unassigned_bytes": unassigned,
        "layers": {str(layer): bytes_by_layer[layer] for layer in sorted(bytes_by_layer)},
        "candidate_plans": plans,
        "tensors": class_rows,
    }


def markdown_report(ledger: dict[str, Any]) -> str:
    model = ledger["model"]
    lines = [
        "# Q38 artifact ledger",
        "",
        f"- Architecture: `{model['architecture']}`",
        f"- Shards/tensors: {model['shard_count']} / {model['tensor_count']}",
        f"- Tensor payload: {gib(model['tensor_payload_bytes']):.3f} GiB",
        f"- Layout SHA-256: `{model['layout_sha256']}`",
        f"- Context admission model: {model['context_limit']:,} tokens",
        "",
        "## Tensor storage",
        "",
        "| Type | Tensors | GiB |",
        "|---|---:|---:|",
    ]
    for name, row in ledger["by_type"].items():
        lines.append(f"| {name} | {row['tensors']} | {gib(row['bytes']):.3f} |")
    lines.extend(
        [
            "",
            "## Candidate contiguous cuts",
            "",
            "Weights are current GGUF payload bytes. QSA state adds BF16 main K/V "
            "and BF16 compressed index only; workspace, graphs and bounded recurrent "
            "state still require the runtime peak planner.",
            "",
            "| Cut | Stage 0 weights | Stage 1 weights | QSA layers | S0 weights+QSA | S1 weights+QSA |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for plan in ledger["candidate_plans"]:
        lines.append(
            f"| {plan['cut']}/{model['layer_count'] - plan['cut']} | "
            f"{gib(plan['stage0_weight_bytes']):.3f} GiB | "
            f"{gib(plan['stage1_weight_bytes']):.3f} GiB | "
            f"{plan['stage0_qsa_layers']}/{plan['stage1_qsa_layers']} | "
            f"{gib(plan['stage0_weight_plus_qsa_bytes']):.3f} GiB | "
            f"{gib(plan['stage1_weight_plus_qsa_bytes']):.3f} GiB |"
        )
    lines.append("")
    if ledger["unassigned_bytes"]:
        lines.append(
            f"Unassigned global tensor bytes: {gib(ledger['unassigned_bytes']):.3f} GiB. "
            "These must be classified before a production stage artifact is emitted."
        )
    else:
        lines.append("All tensor payload bytes are assigned to stage0, stage1, MTP or text-only skip.")
    lines.append("")
    return "\n".join(lines)


def write_tensor_indexes(
    ledger: dict[str, Any], shards: list[ShardInfo], cut: int, output_dir: Path
) -> tuple[Path, Path]:
    model = ledger["model"]
    layer_count = int(model["layer_count"])
    if cut <= 0 or cut >= layer_count:
        raise GGUFError(f"selected cut {cut} is outside 1..{layer_count - 1}")
    shard_paths = {shard.path.name: shard.path.resolve() for shard in shards}
    stage_rows: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for tensor in ledger["tensors"]:
        owner = tensor["owner_class"]
        if owner == "vision":
            continue
        if owner == "unassigned":
            raise GGUFError(f"cannot emit index with unassigned tensor {tensor['name']}")
        if owner == "stage0_global":
            stage = 0
        elif owner in ("stage1_global", "mtp"):
            stage = 1
        elif owner == "layer":
            stage = 0 if int(tensor["layer"]) < cut else 1
        else:
            raise GGUFError(f"unknown tensor owner class {owner}")
        row = dict(tensor)
        shard = next(item for item in shards if item.path.name == tensor["shard"])
        row["absolute_offset"] = shard.data_start + int(tensor["offset"])
        row["shard_path"] = str(shard_paths[tensor["shard"]])
        stage_rows[stage].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / "stage0.q38i", output_dir / "stage1.q38i")
    for stage, path in enumerate(paths):
        lines = [
            "Q38_TENSOR_INDEX_V1",
            f"stage={stage}",
            f"cut={cut}",
            f"layout_sha256={model['layout_sha256']}",
        ]
        for tensor in sorted(stage_rows[stage], key=lambda item: item["name"]):
            lines.append(
                "\t".join(
                    [
                        "tensor",
                        tensor["name"],
                        tensor["type"],
                        str(tensor["payload_bytes"]),
                        str(tensor["absolute_offset"]),
                        tensor["shard_path"],
                        ",".join(str(dim) for dim in tensor["dims"]),
                    ]
                )
            )
        path.write_text("\n".join(lines) + "\n")
    return paths


def parse_cuts(value: str) -> list[int]:
    try:
        cuts = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("cuts must be comma-separated integers") from error
    if not cuts:
        raise argparse.ArgumentTypeError("at least one cut is required")
    return cuts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--cuts", type=parse_cuts, default=[23, 24, 25, 26])
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--selected-cut", type=int)
    parser.add_argument("--index-dir", type=Path)
    args = parser.parse_args()

    if args.context <= 0:
        parser.error("--context must be positive")
    paths = discover_shards(args.model)
    shards = [parse_shard(path) for path in paths]
    artifact_manifest = None
    if args.artifact_manifest:
        artifact_manifest = json.loads(args.artifact_manifest.read_text())
    ledger = build_ledger(shards, args.cuts, args.context, artifact_manifest)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(markdown_report(ledger))
    index_paths = None
    if args.selected_cut is not None or args.index_dir is not None:
        if args.selected_cut is None or args.index_dir is None:
            parser.error("--selected-cut and --index-dir must be used together")
        index_paths = write_tensor_indexes(
            ledger, shards, args.selected_cut, args.index_dir
        )
    print(
        json.dumps(
            {
                "json": str(args.json_out),
                "markdown": str(args.markdown_out),
                "shards": len(shards),
                "tensors": ledger["model"]["tensor_count"],
                "payload_bytes": ledger["model"]["tensor_payload_bytes"],
                "layout_sha256": ledger["model"]["layout_sha256"],
                "indexes": [str(path) for path in index_paths] if index_paths else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

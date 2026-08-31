#!/usr/bin/env python3
"""Qwen3.8 official-source ownership and Ampere quantization policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re


W4A16_G128 = "w4a16_sym_g128"
W8A16_G128 = "w8a16_sym_g128"
FP8_E4M3 = "fp8_e4m3fn"
PRESERVE = "preserve"
SKIP = "skip"

_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_MTP_LAYER = re.compile(r"^mtp\.layers\.(\d+)\.(.+)$")
_ROUTED_EXPERT = re.compile(r"^mlp\.experts\.(gate_up_proj|down_proj)$")


@dataclass(frozen=True)
class QuantDecision:
    source_name: str
    canonical_name: str
    owner: str
    format: str
    group_size: int
    reason: str


def _hc_name(prefix: str, suffix: str) -> str | None:
    mapping = {
        "hc_norm.weight": "norm.weight",
        "input_mix_weight_down.weight": "mix_down.weight",
        "input_mix_weight_up.weight": "mix_up.weight",
        "block_inject_weight.weight": "inject.weight",
    }
    tail = mapping.get(suffix)
    return None if tail is None else f"{prefix}.{tail}"


def _layer_canonical(layer: str, remainder: str) -> str:
    for source, target in (
        ("attn_hyper_connection.", "hc_attn"),
        ("mlp_hyper_connection.", "hc_ffn"),
    ):
        if remainder.startswith(source):
            mapped = _hc_name(target, remainder[len(source) :])
            if mapped:
                return f"{layer}.{mapped}"

    exact = {
        "linear_attn.A_log": "linear_attn.a_log",
        "linear_attn.conv1d.weight": "linear_attn.conv.weight",
        "linear_attn.dt_bias": "linear_attn.dt_bias",
        "linear_attn.in_proj_a.weight": "linear_attn.in_a.weight",
        "linear_attn.in_proj_b.weight": "linear_attn.in_b.weight",
        "linear_attn.in_proj_qkv.weight": "linear_attn.qkv.weight",
        "linear_attn.in_proj_z.weight": "linear_attn.z.weight",
        "linear_attn.norm.weight": "linear_attn.norm.weight",
        "linear_attn.out_proj.weight": "linear_attn.out.weight",
        "self_attn.indexer.index_qk_proj.weight": "attn_index_qk.weight",
        "self_attn.indexer.q_layernorm.weight": "attn_index_q_norm.weight",
        "self_attn.indexer.k_layernorm.weight": "attn_index_k_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.experts.gate_up_proj": "ffn_gate_up_exps.weight",
        "mlp.experts.down_proj": "ffn_down_exps.weight",
        "mlp.gate.weight": "ffn_gate_inp.weight",
        "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
        "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
        "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
        "mlp.shared_expert_gate.weight": "ffn_shexp_gate_inp.weight",
        "ple.conv1d.weight": "ple.conv.weight",
        "ple.key_proj.weight": "ple.key.weight",
        "ple.value_proj.weight": "ple.value.weight",
        "ple.norm_conv.weight": "ple.conv_norm.weight",
        "ple.norm_key.weight": "ple.key_norm.weight",
        "ple.norm_query.weight": "ple.query_norm.weight",
        "ple.ple_embedding.layer_multipliers": "ple.layer_multipliers",
        "ple.ple_embedding.ngram_heads_offsets": "ple.head_offsets",
        "ple.ple_embedding.ngram_heads_vocab_sizes": "ple.head_vocab_sizes",
    }
    if remainder in exact:
        return f"{layer}.{exact[remainder]}"
    shard = re.fullmatch(
        r"ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight",
        remainder,
    )
    if shard:
        return f"ple.table.part.{int(shard.group(1)):03d}"
    raise ValueError(f"unmapped official Qwen3.8 layer tensor: {remainder}")


def canonical_name(source_name: str) -> str:
    if source_name == "model.language_model.embed_tokens.weight":
        return "token_embd.weight"
    if source_name == "lm_head.weight":
        return "output.weight"
    prefix = "model.language_model.hyper_connection_mixer."
    if source_name.startswith(prefix):
        result = _hc_name("hc_input", source_name[len(prefix) :])
        if result:
            return result
    match = _LAYER.match(source_name)
    if match:
        return _layer_canonical(f"blk.{int(match.group(1))}", match.group(2))
    match = _MTP_LAYER.match(source_name)
    if match:
        return _layer_canonical(f"mtp.blk.{int(match.group(1))}", match.group(2))
    mtp_exact = {
        "mtp.fc_embedding.weight": "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight": "mtp.fc_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight": "mtp.fc_embedding_norm.weight",
        "mtp.pre_fc_norm_hidden.weight": "mtp.fc_hidden_norm.weight",
    }
    if source_name in mtp_exact:
        return mtp_exact[source_name]
    mtp_hc = "mtp.hyper_connection_mixer."
    if source_name.startswith(mtp_hc):
        result = _hc_name("mtp.hc_input", source_name[len(mtp_hc) :])
        if result:
            return result
    if source_name.startswith("model.visual."):
        return "vision." + source_name[len("model.visual.") :]
    raise ValueError(f"unmapped official Qwen3.8 tensor: {source_name}")


def owner_for(source_name: str, cut: int) -> str:
    if not 0 < cut < 48:
        raise ValueError("cut must be in 1..47")
    if source_name.startswith("model.visual."):
        return "skip"
    match = _LAYER.match(source_name)
    if match:
        return "stage0" if int(match.group(1)) < cut else "stage1"
    if source_name.startswith("mtp.") or source_name == "lm_head.weight":
        return "stage1"
    if source_name.startswith("model.language_model.hyper_connection_mixer."):
        # This is the final mixer after all 48 base layers, not the input
        # embedding mixer. It consumes stage1's last boundary locally.
        return "stage1"
    if source_name.startswith("model.language_model."):
        return "stage0"
    raise ValueError(f"cannot assign stage owner: {source_name}")


def decide(source_name: str, cut: int = 25) -> QuantDecision:
    owner = owner_for(source_name, cut)
    canonical = canonical_name(source_name)
    if owner == "skip":
        return QuantDecision(source_name, canonical, owner, SKIP, 0, "text-only")

    if ".ple.ple_embedding.ngram_embedding.shard_" in source_name:
        return QuantDecision(
            source_name, canonical, owner, FP8_E4M3, 0, "SSD PLE bandwidth"
        )

    match = _LAYER.match(source_name)
    if match and _ROUTED_EXPERT.match(match.group(2)):
        layer = int(match.group(1))
        if layer in (0, 1, 46, 47):
            return QuantDecision(
                source_name,
                canonical,
                owner,
                W8A16_G128,
                128,
                "quality-sensitive edge routed expert",
            )
        return QuantDecision(
            source_name,
            canonical,
            owner,
            W4A16_G128,
            128,
            "interior routed expert",
        )

    match = _MTP_LAYER.match(source_name)
    if match and _ROUTED_EXPERT.match(match.group(2)):
        return QuantDecision(
            source_name, canonical, owner, W8A16_G128, 128, "MTP quality"
        )

    # Small recurrent/control tensors and hyper-connection mixers keep their
    # source BF16/F32/I64 dtype.  These are sensitive and not the memory driver.
    preserve_markers = (
        "hyper_connection",
        ".hc_norm.",
        ".norm.",
        ".ple.norm_",
        "_norm.",
        "_norm_",
        "layernorm.weight",
        ".A_log",
        ".dt_bias",
        ".conv1d.weight",
        ".mlp.gate.weight",
        ".shared_expert_gate.weight",
        "layer_multipliers",
        "ngram_heads_offsets",
        "ngram_heads_vocab_sizes",
    )
    if any(marker in source_name for marker in preserve_markers):
        return QuantDecision(
            source_name, canonical, owner, PRESERVE, 0, "critical/control tensor"
        )

    if source_name.endswith(".weight") or source_name.endswith("_proj"):
        return QuantDecision(
            source_name,
            canonical,
            owner,
            W8A16_G128,
            128,
            "always-active Ampere matrix",
        )
    raise ValueError(f"no quantization policy for {source_name}")


def policy_identity(cut: int = 25) -> str:
    policy = {
        "schema": "Q38_AMPERE_QUANT_POLICY_V5",
        "cut": cut,
        # The target embedding normally belongs to stage0, but the embedded
        # MTP head runs on stage1 and consumes token embeddings locally.  The
        # artifact compiler therefore materializes a second, identical device
        # copy in stage1.  Keep this in the policy identity so artifacts made
        # before and after the replication rule can never be paired.
        "stage1_mtp_embedding_copy": True,
        "stage1_final_hyper_mixer": True,
        "interior_routed": W4A16_G128,
        "edge_routed": W8A16_G128,
        "mtp": W8A16_G128,
        "always_active": W8A16_G128,
        "critical": PRESERVE,
        "ple": "fp8_e4m3fn_row_bf16_scale",
        "ple_norms": PRESERVE,
        "vision": SKIP,
    }
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

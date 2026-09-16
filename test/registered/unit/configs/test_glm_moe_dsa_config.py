import json

import torch
import transformers.configuration_utils as configuration_utils

from sglang.srt.utils.hf_transformers.config import (
    _ensure_glm_moe_dsa_layer_type_compatibility,
)
from sglang.srt.utils.hf_transformers.tokenizer import (
    _auto_tokenizer_from_pretrained,
)
from sglang.srt.configs.model_config import (
    is_glm_moe_dsa_w4a16_slimquant_compatible,
    should_preserve_explicit_unquantized_draft,
)
from sglang.srt.layers.moe.ep_moe.layer import _dequantize_deepep_hidden_states
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _convert_compressed_tensors_w4a16_for_slimquant,
)


def test_glm_moe_dsa_allows_deepseek_sparse_attention(tmp_path, monkeypatch):
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "layer_types": ["deepseek_sparse_attention"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        configuration_utils,
        "ALLOWED_LAYER_TYPES",
        tuple(
            item
            for item in configuration_utils.ALLOWED_LAYER_TYPES
            if item != "deepseek_sparse_attention"
        ),
    )

    _ensure_glm_moe_dsa_layer_type_compatibility(tmp_path, revision=None)

    assert "deepseek_sparse_attention" in configuration_utils.ALLOWED_LAYER_TYPES


def test_non_glm_config_does_not_change_allowed_layer_types(tmp_path, monkeypatch):
    config = {
        "architectures": ["OtherForCausalLM"],
        "model_type": "other",
        "layer_types": ["deepseek_sparse_attention"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    allowed_layer_types = configuration_utils.ALLOWED_LAYER_TYPES
    monkeypatch.setattr(
        configuration_utils, "ALLOWED_LAYER_TYPES", allowed_layer_types
    )

    _ensure_glm_moe_dsa_layer_type_compatibility(tmp_path, revision=None)

    assert configuration_utils.ALLOWED_LAYER_TYPES == allowed_layer_types


def test_tokenizer_load_applies_glm_layer_type_compatibility(tmp_path, monkeypatch):
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "layer_types": ["deepseek_sparse_attention"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        configuration_utils,
        "ALLOWED_LAYER_TYPES",
        tuple(
            item
            for item in configuration_utils.ALLOWED_LAYER_TYPES
            if item != "deepseek_sparse_attention"
        ),
    )

    sentinel = object()

    def fake_from_pretrained(*args, **kwargs):
        assert "deepseek_sparse_attention" in configuration_utils.ALLOWED_LAYER_TYPES
        return sentinel

    monkeypatch.setattr(
        "sglang.srt.utils.hf_transformers.tokenizer.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    assert _auto_tokenizer_from_pretrained(tmp_path) is sentinel


def test_glm_w4a16_checkpoint_is_slimquant_compatible():
    hf_config = {"architectures": ["GlmMoeDsaForCausalLM"]}
    quant_config = {
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "weights": {
                    "type": "int",
                    "num_bits": 4,
                    "strategy": "channel",
                    "group_size": -1,
                    "symmetric": True,
                    "dynamic": False,
                },
            }
        },
    }

    assert is_glm_moe_dsa_w4a16_slimquant_compatible(hf_config, quant_config)

    quant_config["config_groups"]["group_0"]["input_activations"] = {
        "num_bits": 8
    }
    assert not is_glm_moe_dsa_w4a16_slimquant_compatible(
        hf_config, quant_config
    )


def test_explicit_unquantized_draft_survives_checkpoint_detection():
    assert should_preserve_explicit_unquantized_draft(
        is_draft_model=True,
        is_draft_quantization_explicit=True,
        quantization=None,
    )
    assert not should_preserve_explicit_unquantized_draft(
        is_draft_model=False,
        is_draft_quantization_explicit=True,
        quantization=None,
    )
    assert not should_preserve_explicit_unquantized_draft(
        is_draft_model=True,
        is_draft_quantization_explicit=False,
        quantization=None,
    )
    assert not should_preserve_explicit_unquantized_draft(
        is_draft_model=True,
        is_draft_quantization_explicit=True,
        quantization="slimquant_w4a8_marlin",
    )


def test_deepep_int8_transport_is_dequantized_for_unquantized_moe():
    hidden_states = torch.tensor([[[2, -3], [4, 5]]], dtype=torch.int8)
    scales = torch.tensor([[0.5, 0.25]], dtype=torch.float32)

    actual = _dequantize_deepep_hidden_states(
        hidden_states, scales, torch.bfloat16
    )

    expected = torch.tensor([[[1.0, -1.5], [1.0, 1.25]]], dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected)


def test_compressed_tensors_w4a16_is_converted_to_slimquant_layout():
    unsigned_nibbles = [0, 1, 7, 8, 9, 15, 11, 4]
    packed = sum(value << (4 * index) for index, value in enumerate(unsigned_nibbles))
    packed = torch.tensor([[packed]], dtype=torch.int64).to(torch.int32)

    name, actual = _convert_compressed_tensors_w4a16_for_slimquant(
        "model.layers.3.mlp.experts.0.gate_proj.weight_packed", packed
    )

    assert name.endswith("gate_proj.weight")
    torch.testing.assert_close(
        actual, torch.tensor([[-119, -16, 23, 60]], dtype=torch.int8)
    )

    scale_name, scale = _convert_compressed_tensors_w4a16_for_slimquant(
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale",
        torch.tensor([[0.016]], dtype=torch.float32),
    )
    assert scale_name.endswith("gate_proj.weight_scale")
    torch.testing.assert_close(scale, torch.tensor([[0.001]], dtype=torch.float32))


def test_compressed_tensors_w4a16_shape_metadata_is_skipped():
    name, value = _convert_compressed_tensors_w4a16_for_slimquant(
        "model.layers.3.mlp.experts.0.gate_proj.weight_shape",
        torch.tensor([2048, 6144]),
    )
    assert name is None
    assert value is None

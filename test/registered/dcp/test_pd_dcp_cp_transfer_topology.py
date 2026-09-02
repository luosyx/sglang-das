import pytest

from sglang.srt.disaggregation.common.conn import (
    PrefillServerInfo,
    validate_pd_dcp_prefill_topology,
)


def test_prefill_info_round_trips_all_cp_transfer_capability():
    info = PrefillServerInfo(
        attn_tp_size=8,
        attn_cp_size=8,
        dp_size=1,
        pp_size=2,
        page_size=64,
        kv_cache_dtype="fp8_e4m3",
        follow_bootstrap_room=True,
        enable_all_cp_ranks_for_transfer=True,
    )

    assert info.enable_all_cp_ranks_for_transfer is True


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_pd_dcp_accepts_prefill_cp_when_all_ranks_transfer(dcp_size):
    validate_pd_dcp_prefill_topology(
        decode_dcp_size=dcp_size,
        decode_is_mla=True,
        prefill_attn_cp_size=8,
        prefill_all_cp_ranks_transfer=True,
        prefill_dsa_cache_layer_split=False,
    )


def test_pd_dcp_rejects_partial_prefill_cp_transfer():
    with pytest.raises(RuntimeError, match="ALL_CP_RANKS_TRANSFER"):
        validate_pd_dcp_prefill_topology(
            decode_dcp_size=8,
            decode_is_mla=True,
            prefill_attn_cp_size=8,
            prefill_all_cp_ranks_transfer=False,
            prefill_dsa_cache_layer_split=False,
        )


def test_pd_dcp_still_requires_mla_pool():
    with pytest.raises(RuntimeError, match="MLA or hybrid-MLA"):
        validate_pd_dcp_prefill_topology(
            decode_dcp_size=8,
            decode_is_mla=False,
            prefill_attn_cp_size=1,
            prefill_all_cp_ranks_transfer=False,
            prefill_dsa_cache_layer_split=False,
        )

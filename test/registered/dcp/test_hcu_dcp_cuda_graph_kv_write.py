import unittest

import torch

from sglang.srt.mem_cache.memory_pool import (
    _remap_hcu_dcp_graph_kv_write_locs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHCUDCPCudaGraphKVWrite(unittest.TestCase):
    def test_non_owner_rows_are_redirected_to_distinct_padding_slots(self):
        loc = torch.arange(8, dtype=torch.int64)

        mapped = _remap_hcu_dcp_graph_kv_write_locs(
            loc,
            dcp_size=8,
            dcp_rank=3,
            pool_size=100,
            page_size=64,
        )

        self.assertEqual(mapped[3].item(), 0)
        self.assertEqual(mapped[:3].tolist(), [100, 101, 102])
        self.assertEqual(mapped[4:].tolist(), [104, 105, 106, 107])
        self.assertEqual(mapped.unique().numel(), loc.numel())
        self.assertLess(mapped.max().item(), 100 + 64)

    def test_capture_batch_must_fit_reserved_padding_page(self):
        with self.assertRaisesRegex(AssertionError, "not to exceed page size"):
            _remap_hcu_dcp_graph_kv_write_locs(
                torch.arange(65),
                dcp_size=8,
                dcp_rank=0,
                pool_size=100,
                page_size=64,
            )


if __name__ == "__main__":
    unittest.main()

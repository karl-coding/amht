import unittest

from train.train import (
    disable_memory_for_batch,
    disable_router_attention_for_batch,
    disable_router_aux_for_batch,
    disable_router_straight_through_for_batch,
)


class TrainStateTrackingRuntimeTests(unittest.TestCase):
    def test_state_tracking_defaults_preserve_existing_disables(self) -> None:
        cfg = {}
        batch_sources = {"state_tracking": 1}

        self.assertTrue(disable_router_aux_for_batch(cfg, "mixed", batch_sources))
        self.assertTrue(disable_router_straight_through_for_batch(cfg, "mixed", batch_sources))
        self.assertTrue(disable_router_attention_for_batch(cfg, "mixed", batch_sources))
        self.assertTrue(disable_memory_for_batch(cfg, "mixed", batch_sources))

    def test_state_tracking_runtime_can_keep_memory_enabled(self) -> None:
        cfg = {
            "data": {
                "state_tracking_runtime": {
                    "disable_memory": False,
                }
            }
        }

        self.assertFalse(disable_memory_for_batch(cfg, "mixed", {"state_tracking": 1}))
        self.assertFalse(disable_memory_for_batch(cfg, "mixed", {"retrieval": 1}))

    def test_state_tracking_runtime_can_override_all_disables(self) -> None:
        cfg = {
            "data": {
                "state_tracking_runtime": {
                    "disable_router_aux": False,
                    "disable_router_straight_through": False,
                    "disable_router_attention": False,
                    "disable_memory": False,
                }
            }
        }
        batch_sources = {"state_tracking": 1}

        self.assertFalse(disable_router_aux_for_batch(cfg, "mixed", batch_sources))
        self.assertFalse(disable_router_straight_through_for_batch(cfg, "mixed", batch_sources))
        self.assertFalse(disable_router_attention_for_batch(cfg, "mixed", batch_sources))
        self.assertFalse(disable_memory_for_batch(cfg, "mixed", batch_sources))


if __name__ == "__main__":
    unittest.main()

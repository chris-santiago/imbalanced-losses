"""
Tests for _MemoryQueue.

Coverage
--------
- Init: buffer registration, shapes, dtypes, ignore_index, size property
- enqueue: advances pointer, stores values, wrap-around, batch >= queue_size,
           no-op when queue_size=0
- merge: returns inputs unchanged when queue_size=0, concatenates, device/dtype
         cast, unfilled slots carry ignore_index
- reset: clears logits, restores ignore_index targets, zeros pointer,
         no-op when queue_size=0
"""

from __future__ import annotations

import torch

from imbalanced_losses._queue import _MemoryQueue


# ---------------------------------------------------------------------------
# Init / construction
# ---------------------------------------------------------------------------


class TestMemoryQueueInit:
    C = 4

    def test_buffers_registered_positive_queue_size(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        assert hasattr(q, "_q_logits")
        assert hasattr(q, "_q_targets")
        assert hasattr(q, "_q_ptr")

    def test_logits_buffer_shape(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        assert q._q_logits.shape == (16, self.C)

    def test_targets_buffer_shape(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        assert q._q_targets.shape == (16,)

    def test_ptr_buffer_shape(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        assert q._q_ptr.shape == (1,)

    def test_logits_buffer_initialised_to_zeros(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        assert (q._q_logits == 0).all()

    def test_targets_buffer_initialised_to_ignore_index(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C, ignore_index=-100)
        assert (q._q_targets == -100).all()

    def test_custom_ignore_index_stored_in_targets(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C, ignore_index=999)
        assert (q._q_targets == 999).all()

    def test_ptr_initialised_to_zero(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        assert int(q._q_ptr) == 0

    def test_targets_dtype_is_long(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        assert q._q_targets.dtype == torch.long

    def test_ptr_dtype_is_long(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        assert q._q_ptr.dtype == torch.long

    def test_no_buffers_when_queue_size_zero(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        assert not hasattr(q, "_q_logits")
        assert not hasattr(q, "_q_targets")
        assert not hasattr(q, "_q_ptr")

    def test_size_property_equals_queue_size(self):
        q = _MemoryQueue(queue_size=32, num_classes=self.C)
        assert q.size == 32

    def test_size_property_zero(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        assert q.size == 0

    def test_is_nn_module(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        assert isinstance(q, torch.nn.Module)

    def test_buffers_appear_in_state_dict(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        sd = q.state_dict()
        assert "_q_logits" in sd
        assert "_q_targets" in sd
        assert "_q_ptr" in sd


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------


class TestEnqueue:
    C = 3

    def test_advances_pointer(self):
        q = _MemoryQueue(queue_size=32, num_classes=self.C)
        q.enqueue(torch.zeros(8, self.C), torch.zeros(8, dtype=torch.long))
        assert int(q._q_ptr) == 8

    def test_stores_logit_values(self):
        q = _MemoryQueue(queue_size=32, num_classes=self.C)
        logits = torch.ones(4, self.C) * 7.0
        targets = torch.tensor([0, 1, 2, 0])
        q.enqueue(logits, targets)
        assert torch.allclose(q._q_logits[:4], logits)

    def test_stores_target_values(self):
        q = _MemoryQueue(queue_size=32, num_classes=self.C)
        targets = torch.tensor([0, 1, 2, 0])
        q.enqueue(torch.ones(4, self.C), targets)
        assert (q._q_targets[:4] == targets).all()

    def test_sequential_enqueue_advances_pointer(self):
        q = _MemoryQueue(queue_size=32, num_classes=self.C)
        q.enqueue(torch.zeros(5, self.C), torch.zeros(5, dtype=torch.long))
        q.enqueue(torch.zeros(3, self.C), torch.zeros(3, dtype=torch.long))
        assert int(q._q_ptr) == 8

    def test_wrap_around_pointer(self):
        # Fill 6, then add 4 more → pointer wraps to (6+4) % 8 = 2
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        q.enqueue(torch.zeros(6, self.C), torch.zeros(6, dtype=torch.long))
        q.enqueue(torch.ones(4, self.C) * 9.0, torch.ones(4, dtype=torch.long))
        assert int(q._q_ptr) == 2

    def test_wrap_around_stores_head_values(self):
        # The 2 rows that wrapped should land at indices 0 and 1
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        q.enqueue(torch.zeros(6, self.C), torch.zeros(6, dtype=torch.long))
        fill = torch.ones(4, self.C) * 9.0
        q.enqueue(fill, torch.ones(4, dtype=torch.long))
        assert torch.allclose(q._q_logits[0], torch.ones(self.C) * 9.0)
        assert torch.allclose(q._q_logits[1], torch.ones(self.C) * 9.0)

    def test_wrap_around_stores_tail_values(self):
        # Rows that didn't wrap should sit at [6] and [7]
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        q.enqueue(torch.zeros(6, self.C), torch.zeros(6, dtype=torch.long))
        fill = torch.ones(4, self.C) * 9.0
        q.enqueue(fill, torch.ones(4, dtype=torch.long))
        assert torch.allclose(q._q_logits[6], torch.ones(self.C) * 9.0)
        assert torch.allclose(q._q_logits[7], torch.ones(self.C) * 9.0)

    def test_batch_larger_than_queue_replaces_buffer(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        big = torch.arange(32 * self.C, dtype=torch.float).view(32, self.C)
        q.enqueue(big, torch.zeros(32, dtype=torch.long))
        assert torch.allclose(q._q_logits, big[-8:])

    def test_batch_larger_than_queue_resets_pointer(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        big = torch.arange(32 * self.C, dtype=torch.float).view(32, self.C)
        q.enqueue(big, torch.zeros(32, dtype=torch.long))
        assert int(q._q_ptr) == 0

    def test_batch_equal_to_queue_size_replaces_buffer(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        exact = torch.ones(8, self.C) * 5.0
        q.enqueue(exact, torch.ones(8, dtype=torch.long))
        assert torch.allclose(q._q_logits, exact)
        assert int(q._q_ptr) == 0

    def test_no_op_when_queue_size_zero(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        # Must not raise
        q.enqueue(torch.zeros(4, self.C), torch.zeros(4, dtype=torch.long))

    def test_enqueue_detaches_logits(self):
        # Queue should store values without holding grad_fn references
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        logits = torch.randn(4, self.C, requires_grad=True)
        targets = torch.zeros(4, dtype=torch.long)
        q.enqueue(logits, targets)
        assert q._q_logits[:4].grad_fn is None


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


class TestMerge:
    C = 3

    def test_returns_inputs_unchanged_when_queue_size_zero(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        logits = torch.randn(4, self.C)
        targets = torch.zeros(4, dtype=torch.long)
        out_l, out_t = q.merge(logits, targets)
        assert out_l is logits
        assert out_t is targets

    def test_output_logits_shape(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        logits = torch.randn(4, self.C)
        targets = torch.zeros(4, dtype=torch.long)
        out_l, _ = q.merge(logits, targets)
        assert out_l.shape == (4 + 8, self.C)

    def test_output_targets_shape(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        logits = torch.randn(4, self.C)
        targets = torch.zeros(4, dtype=torch.long)
        _, out_t = q.merge(logits, targets)
        assert out_t.shape == (4 + 8,)

    def test_live_batch_comes_first(self):
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        logits = torch.ones(3, self.C) * 7.0
        targets = torch.tensor([1, 2, 0])
        out_l, out_t = q.merge(logits, targets)
        assert torch.allclose(out_l[:3], logits)
        assert (out_t[:3] == targets).all()

    def test_queue_contents_follow_live_batch(self):
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        # Fill queue with known values
        fill = torch.ones(4, self.C) * 3.0
        fill_targets = torch.tensor([0, 1, 2, 0])
        q.enqueue(fill, fill_targets)

        logits = torch.zeros(2, self.C)
        targets = torch.zeros(2, dtype=torch.long)
        out_l, out_t = q.merge(logits, targets)
        assert torch.allclose(out_l[2:], fill)
        assert (out_t[2:] == fill_targets).all()

    def test_unfilled_slots_carry_ignore_index(self):
        # Fresh queue: all target slots should be ignore_index
        q = _MemoryQueue(queue_size=8, num_classes=self.C, ignore_index=-100)
        logits = torch.randn(4, self.C)
        targets = torch.zeros(4, dtype=torch.long)
        _, out_t = q.merge(logits, targets)
        # The queue portion is the last 8 entries
        assert (out_t[4:] == -100).all()

    def test_dtype_cast_in_merge(self):
        # Queue was constructed in float32; live batch is float16 — queue should be cast
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        logits = torch.randn(2, self.C).to(torch.float16)
        targets = torch.zeros(2, dtype=torch.long)
        out_l, _ = q.merge(logits, targets)
        assert out_l.dtype == torch.float16

    def test_merge_concatenates_correctly_after_enqueue(self):
        torch.manual_seed(42)
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        batch1 = torch.randn(5, self.C)
        tgt1 = torch.randint(0, self.C, (5,))
        q.enqueue(batch1, tgt1)

        batch2 = torch.randn(3, self.C)
        tgt2 = torch.randint(0, self.C, (3,))
        out_l, out_t = q.merge(batch2, tgt2)
        # Total rows: 3 live + 8 queue
        assert out_l.shape[0] == 11
        assert out_t.shape[0] == 11


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    C = 3

    def test_zeroes_logits(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        q.enqueue(torch.ones(8, self.C), torch.ones(8, dtype=torch.long))
        q.reset()
        assert (q._q_logits == 0).all()

    def test_restores_ignore_index_in_targets(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C, ignore_index=-100)
        q.enqueue(torch.ones(8, self.C), torch.ones(8, dtype=torch.long))
        q.reset()
        assert (q._q_targets == -100).all()

    def test_zeros_pointer(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        q.enqueue(torch.ones(8, self.C), torch.ones(8, dtype=torch.long))
        q.reset()
        assert int(q._q_ptr) == 0

    def test_pointer_zeroed_after_partial_fill(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        q.enqueue(torch.zeros(5, self.C), torch.zeros(5, dtype=torch.long))
        assert int(q._q_ptr) == 5
        q.reset()
        assert int(q._q_ptr) == 0

    def test_no_op_when_queue_size_zero(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        q.reset()  # must not raise

    def test_enqueue_after_reset_starts_at_zero(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        q.enqueue(torch.ones(10, self.C) * 5.0, torch.ones(10, dtype=torch.long))
        q.reset()
        new_logits = torch.ones(3, self.C) * 9.0
        q.enqueue(new_logits, torch.zeros(3, dtype=torch.long))
        assert torch.allclose(q._q_logits[:3], new_logits)
        assert int(q._q_ptr) == 3

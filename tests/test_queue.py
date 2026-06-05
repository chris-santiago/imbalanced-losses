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


# ---------------------------------------------------------------------------
# _q_iid buffer: init
# ---------------------------------------------------------------------------


class TestIidBufferInit:
    C = 3

    def test_iid_buffer_registered_positive_queue_size(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        assert hasattr(q, "_q_iid")

    def test_iid_buffer_shape(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        assert q._q_iid.shape == (16,)

    def test_iid_buffer_dtype_bool(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        assert q._q_iid.dtype == torch.bool

    def test_iid_buffer_initialised_to_true(self):
        # Unfilled slots default to True (treated as iid for legacy compat).
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        assert q._q_iid.all()

    def test_no_iid_buffer_when_queue_size_zero(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        assert not hasattr(q, "_q_iid")

    def test_iid_buffer_appears_in_state_dict(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        assert "_q_iid" in q.state_dict()


# ---------------------------------------------------------------------------
# _q_iid buffer: enqueue
# ---------------------------------------------------------------------------


class TestEnqueueIid:
    C = 3

    def test_enqueue_explicit_false_flags_stored(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        iid = torch.tensor([True, False, True, False])
        q.enqueue(torch.zeros(4, self.C), torch.zeros(4, dtype=torch.long), is_iid=iid)
        assert (q._q_iid[:4] == iid).all()

    def test_enqueue_none_is_iid_treated_as_all_true(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        q.enqueue(torch.zeros(4, self.C), torch.zeros(4, dtype=torch.long), is_iid=None)
        assert q._q_iid[:4].all()

    def test_enqueue_none_equals_explicit_all_true(self):
        # Both code paths must produce identical buffer contents.
        q1 = _MemoryQueue(queue_size=8, num_classes=self.C)
        q2 = _MemoryQueue(queue_size=8, num_classes=self.C)
        logits = torch.randn(4, self.C)
        targets = torch.zeros(4, dtype=torch.long)
        explicit_true = torch.ones(4, dtype=torch.bool)
        q1.enqueue(logits, targets, is_iid=None)
        q2.enqueue(logits, targets, is_iid=explicit_true)
        assert (q1._q_iid == q2._q_iid).all()

    def test_enqueue_iid_advances_pointer_same_as_before(self):
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        q.enqueue(torch.zeros(5, self.C), torch.zeros(5, dtype=torch.long),
                  is_iid=torch.ones(5, dtype=torch.bool))
        assert int(q._q_ptr) == 5

    def test_enqueue_wrap_around_preserves_iid_alignment(self):
        # Fill 6 rows, then wrap 4 more.  The 2 wrapped rows land at [0],[1].
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        q.enqueue(torch.zeros(6, self.C), torch.zeros(6, dtype=torch.long),
                  is_iid=torch.ones(6, dtype=torch.bool))
        # Second batch: alternating True/False
        iid2 = torch.tensor([True, False, True, False])
        q.enqueue(torch.ones(4, self.C), torch.ones(4, dtype=torch.long), is_iid=iid2)
        # First two of iid2 wrapped to indices 0,1 (those are iid2[2],iid2[3])
        assert bool(q._q_iid[0]) == bool(iid2[2])
        assert bool(q._q_iid[1]) == bool(iid2[3])
        # Tail (indices 6,7) holds iid2[0],iid2[1]
        assert bool(q._q_iid[6]) == bool(iid2[0])
        assert bool(q._q_iid[7]) == bool(iid2[1])

    def test_enqueue_replace_wholesale_preserves_iid(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        big_iid = torch.tensor([True, False, True, False, True, False, True, False,
                                 True, False, True, False])  # 12 rows
        q.enqueue(torch.zeros(12, self.C), torch.zeros(12, dtype=torch.long), is_iid=big_iid)
        # Only the last 8 rows of big_iid should be stored.
        assert (q._q_iid == big_iid[-8:]).all()
        assert int(q._q_ptr) == 0

    def test_enqueue_no_op_queue_size_zero_with_iid(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        # Must not raise even when is_iid is provided.
        q.enqueue(torch.zeros(4, self.C), torch.zeros(4, dtype=torch.long),
                  is_iid=torch.ones(4, dtype=torch.bool))

    def test_enqueue_iid_detached(self):
        # _q_iid must not hold grad_fn references.
        q = _MemoryQueue(queue_size=16, num_classes=self.C)
        logits = torch.randn(4, self.C, requires_grad=True)
        targets = torch.zeros(4, dtype=torch.long)
        iid = torch.ones(4, dtype=torch.bool)
        q.enqueue(logits, targets, is_iid=iid)
        # bool tensors cannot carry grad_fn, but the buffer should not be a
        # float tensor with grad — just verify no AttributeError and it is bool.
        assert q._q_iid[:4].dtype == torch.bool


# ---------------------------------------------------------------------------
# _q_iid buffer: merge
# ---------------------------------------------------------------------------


class TestMergeIid:
    C = 3

    def test_return_iid_false_is_2tuple(self):
        # Default (return_iid=False) must return exactly 2 elements — backward compat.
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        result = q.merge(torch.randn(2, self.C), torch.zeros(2, dtype=torch.long))
        assert len(result) == 2

    def test_return_iid_true_is_3tuple(self):
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        result = q.merge(torch.randn(2, self.C), torch.zeros(2, dtype=torch.long),
                         return_iid=True)
        assert len(result) == 3

    def test_return_iid_false_identical_to_pre_change(self):
        # Shape and values of the 2-tuple must be unchanged.
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        logits = torch.randn(2, self.C)
        targets = torch.tensor([0, 1])
        out_l, out_t = q.merge(logits, targets)
        assert out_l.shape == (2 + 4, self.C)
        assert out_t.shape == (2 + 4,)

    def test_iid_shape_in_3tuple(self):
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        _, _, out_iid = q.merge(torch.randn(2, self.C), torch.zeros(2, dtype=torch.long),
                                return_iid=True)
        assert out_iid.shape == (2 + 4,)

    def test_iid_dtype_bool(self):
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        _, _, out_iid = q.merge(torch.randn(2, self.C), torch.zeros(2, dtype=torch.long),
                                return_iid=True)
        assert out_iid.dtype == torch.bool

    def test_iid_none_in_merge_equals_all_true(self):
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        logits = torch.randn(2, self.C)
        targets = torch.zeros(2, dtype=torch.long)
        # Explicit all-True vs None should produce the same iid tensor.
        _, _, iid_none = q.merge(logits, targets, is_iid=None, return_iid=True)
        _, _, iid_true = q.merge(logits, targets,
                                 is_iid=torch.ones(2, dtype=torch.bool), return_iid=True)
        assert (iid_none == iid_true).all()

    def test_live_batch_iid_comes_first(self):
        # Live-batch flags occupy the first N positions.
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        live_iid = torch.tensor([True, False])
        _, _, out_iid = q.merge(torch.randn(2, self.C), torch.zeros(2, dtype=torch.long),
                                is_iid=live_iid, return_iid=True)
        assert (out_iid[:2] == live_iid).all()

    def test_queue_iid_follows_live_batch(self):
        # Queue-stored flags occupy positions [N:].
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        stored_iid = torch.tensor([True, False, True, False])
        q.enqueue(torch.zeros(4, self.C), torch.zeros(4, dtype=torch.long), is_iid=stored_iid)
        _, _, out_iid = q.merge(torch.randn(2, self.C), torch.zeros(2, dtype=torch.long),
                                return_iid=True)
        assert (out_iid[2:] == stored_iid).all()

    def test_iid_row_alignment_with_logits(self):
        # After enqueue+merge, out_iid[k] must correspond to the same row as
        # out_logits[k] and out_targets[k].
        q = _MemoryQueue(queue_size=4, num_classes=self.C)
        stored_logits = torch.arange(4 * self.C, dtype=torch.float).view(4, self.C)
        stored_targets = torch.tensor([0, 1, 2, 0])
        stored_iid = torch.tensor([True, False, True, False])
        q.enqueue(stored_logits, stored_targets, is_iid=stored_iid)

        live_logits = torch.zeros(2, self.C)
        live_targets = torch.tensor([1, 2])
        live_iid = torch.tensor([False, True])
        out_l, out_t, out_iid = q.merge(live_logits, live_targets, is_iid=live_iid,
                                        return_iid=True)
        # Live rows at [0],[1]: check flag, target, logit row alignment.
        assert bool(out_iid[0]) == False
        assert bool(out_iid[1]) == True
        assert int(out_t[0]) == 1
        assert int(out_t[1]) == 2
        # Queue rows at [2:6]: must match stored_iid exactly.
        assert (out_iid[2:] == stored_iid).all()
        assert (out_t[2:] == stored_targets).all()

    def test_merge_queue_size_zero_return_iid_false(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        logits = torch.randn(4, self.C)
        targets = torch.zeros(4, dtype=torch.long)
        result = q.merge(logits, targets)
        assert len(result) == 2
        out_l, out_t = result
        assert out_l is logits
        assert out_t is targets

    def test_merge_queue_size_zero_return_iid_true_none(self):
        # When queue_size=0 and is_iid=None, synthesize all-True for the live batch.
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        logits = torch.randn(4, self.C)
        targets = torch.zeros(4, dtype=torch.long)
        out_l, out_t, out_iid = q.merge(logits, targets, return_iid=True)
        assert out_l is logits
        assert out_t is targets
        assert out_iid.shape == (4,)
        assert out_iid.dtype == torch.bool
        assert out_iid.all()

    def test_merge_queue_size_zero_return_iid_true_explicit(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        logits = torch.randn(3, self.C)
        targets = torch.zeros(3, dtype=torch.long)
        live_iid = torch.tensor([True, False, True])
        out_l, out_t, out_iid = q.merge(logits, targets, is_iid=live_iid, return_iid=True)
        assert (out_iid == live_iid).all()


# ---------------------------------------------------------------------------
# _q_iid buffer: reset
# ---------------------------------------------------------------------------


class TestResetIid:
    C = 3

    def test_reset_restores_iid_to_all_true(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        q.enqueue(torch.zeros(4, self.C), torch.zeros(4, dtype=torch.long),
                  is_iid=torch.zeros(4, dtype=torch.bool))
        q.reset()
        assert q._q_iid.all()

    def test_reset_no_op_when_queue_size_zero(self):
        q = _MemoryQueue(queue_size=0, num_classes=self.C)
        q.reset()  # must not raise

    def test_enqueue_after_reset_iid_starts_fresh(self):
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        q.enqueue(torch.zeros(6, self.C), torch.zeros(6, dtype=torch.long),
                  is_iid=torch.zeros(6, dtype=torch.bool))
        q.reset()
        new_iid = torch.tensor([True, False, True])
        q.enqueue(torch.ones(3, self.C), torch.zeros(3, dtype=torch.long), is_iid=new_iid)
        assert (q._q_iid[:3] == new_iid).all()
        # Remaining slots (not yet written) should still be True from reset.
        assert q._q_iid[3:].all()


# ---------------------------------------------------------------------------
# _q_iid buffer: checkpoint / state_dict
# ---------------------------------------------------------------------------


class TestIidCheckpointCompat:
    C = 3

    def test_state_dict_round_trip(self):
        # Save and reload; _q_iid must survive with correct values.
        q = _MemoryQueue(queue_size=8, num_classes=self.C)
        iid = torch.tensor([True, False, True, False, True, True, False, True])
        q.enqueue(torch.randn(8, self.C), torch.zeros(8, dtype=torch.long), is_iid=iid)
        sd = q.state_dict()
        assert "_q_iid" in sd

        q2 = _MemoryQueue(queue_size=8, num_classes=self.C)
        q2.load_state_dict(sd)
        assert (q2._q_iid == iid).all()

    def test_legacy_state_dict_missing_iid_loads_as_all_true(self):
        # Simulate a checkpoint saved before _q_iid was introduced:
        # the key is absent from the state_dict.
        q_legacy = _MemoryQueue(queue_size=8, num_classes=self.C)
        sd = q_legacy.state_dict()
        del sd["_q_iid"]  # remove to simulate legacy checkpoint

        q_new = _MemoryQueue(queue_size=8, num_classes=self.C)
        # Expect no error and _q_iid defaults to all-True.
        q_new.load_state_dict(sd, strict=False)
        assert q_new._q_iid.all()

    def test_legacy_strict_load_injects_all_true_default(self):
        # Even with strict=True, a missing _q_iid should be handled gracefully
        # (injected as all-True in _load_from_state_dict before strict check).
        q_legacy = _MemoryQueue(queue_size=8, num_classes=self.C)
        sd = q_legacy.state_dict()
        del sd["_q_iid"]

        q_new = _MemoryQueue(queue_size=8, num_classes=self.C)
        # _load_from_state_dict injects the default so strict=True doesn't error.
        q_new.load_state_dict(sd, strict=True)
        assert q_new._q_iid.all()

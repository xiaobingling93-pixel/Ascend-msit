import math
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype


def get_slots(bs, block_size, prefill_max_len, is_prefill, block_tables, valid_length_example):
    """get_slots."""
    slot_mapping = []
    for i in range(bs):
        block_table = block_tables[i]
        if is_prefill:
            slots = [block_table[k // block_size] * block_size + k % block_size
                        for k in range(valid_length_example[i])]
            null_slot_idx = -1
            num_elements_to_add = prefill_max_len - valid_length_example[i]
            for _ in range(num_elements_to_add):
                slots.append(null_slot_idx)
        else:
            current_idx = valid_length_example[i] - 1
            slots = [block_table[current_idx // block_size] * block_size + current_idx % block_size]
        slot_mapping = slot_mapping + slots

    return np.array(slot_mapping, copy=False, dtype=np.int32)

def gen_fake_inputs(bs, seq, block_size):
    input_seq_len = seq // 2 + 1
    valid_length_each_example = np.array([input_seq_len])
    prefill_max_len = max(valid_length_each_example)
    required_block_num = math.ceil(input_seq_len / block_size)
    block_tables = np.arange(required_block_num, dtype=np.int32).reshape(bs, -1)
    slot_mapping = get_slots(bs, block_size, prefill_max_len, True, block_tables,
                                            valid_length_each_example)
    input_ids = np.ones(input_seq_len, dtype=np.int64).reshape(bs, -1)

    input_ids = Tensor(input_ids, mstype.int32)
    input_position = Tensor(input_seq_len, mstype.int32)
    init_reset = Tensor([False], mstype.bool_)
    batch_valid_length = Tensor([valid_length_each_example], mstype.int32)
    block_tables = Tensor(block_tables, mstype.int32)
    slot_mapping = Tensor(slot_mapping, mstype.int32)

    return [input_ids, None, input_position, None, None, None, init_reset, batch_valid_length, None, None,
            block_tables, slot_mapping]
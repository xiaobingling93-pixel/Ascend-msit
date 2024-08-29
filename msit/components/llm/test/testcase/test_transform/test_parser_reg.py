from unittest import TestCase

from msit_llm.transform.model_parser.parser import parse_input_names

LLAMA_CASE_01 = """
DecoderModel::~DecoderModel() {}

std::map<std::string, std::vector<std::string>> GetLlamaModelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> llamaInTensorCandiadates = {
        {"default", {
            "input_ids", "input_embedding", "positional_ids", "cosine_table", "sine_table", "attention_mask",
            "block_tables", "slots", "kv_cache_idx", "token_offset", "place_holder", "seq_len", "logits_indices"}
        },
        {"compress_head", {"batch_wins", "ra_seq_len"}},
        {"speculate", {"q_len"}},
        {"lora_common", {"seq_len_cum_sum"}},
        {"lora_per_layer", {
            "qkv_lora_a_0", "qkv_lora_b_0", "qkv_lora_a_1", "qkv_lora_b_1",
            "qkv_lora_a_2", "qkv_lora_b_2", "qkv_dense_lora_a", "qkv_dense_lora_b",
            "mlp_lora_a_0", "mlp_lora_b_0", "mlp_lora_a_1", "mlp_lora_b_1",
            "mlp_down_lora_a", "mlp_down_lora_b"}
        },
    };
    return llamaInTensorCandiadates;
}
"""


LLAMA_CASE_02 = """
enum InTensorId : int {
    // define inTensor
    // idx: 0, input_ids, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_INPUT = 0,
    // idx: 1, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_POSITION_IDS,

    // idx: 2, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_COS_TABLE,
    // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_SIN_TABLE,
    // idx: 4, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    // PA: [maxInputLength, maxInputLength]
    IN_TENSOR_ATTENTION_MASK,
    // idx: 5, shape: [4, 9]; PA所需入参
    IN_TENSOR_BLOCK_TABLES,
    // idx: 6, shape: [seqLen]; PA所需入参
    IN_TENSOR_SLOTS,
    // idx: 7, shape: [1]; FA所需入参
    IN_TENSOR_KV_CACHE_IDX,
    // idx: 8, shape: [batchSize]; FA所需入参
    IN_TENSOR_TOKEN_OFFSET,
    // idx: 9, shape: FA: [batchSize] PA: [4]
    IN_TENSOR_SEQ_LENGTHS,
    // idx: 10, shape: FA: [batchSize]  PA: [4]
    IN_TENSOR_LOGTIS_INDICES,

    IN_PLACEHOLDER,
    IN_TENSOR_Q_LEN,
    IN_TENSOR_MAX,
};
"""


QWEN_CASE_01 = """
enum InTensorId : int {
    IN_TENSOR_INPUT = 0,  // input_ids or word_embedding
    IN_TENSOR_POSITIONIDS,
    IN_TENSOR_COS,  // cos_table or cos_embed
    IN_TENSOR_SIN,  // sin_table or sin_embed
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_KV_CACHE_IDX,
    IN_TENSOR_TOKEN_OFFSET,
    IN_TENSOR_SEQ_LENGTHS,
    IN_TENSOR_LOGTIS_INDICES,
    IN_PLACEHOLDER,
    IN_TENSOR_Q_LEN,
    IN_TENSOR_MAX,
};

enum InternalTensorId : int {
    INTERNAL_HIDDENSTATES = 0,
    INTERNAL_COSEMBED,
    INTERNAL_SINEMBED,
    INTERNAL_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};
"""

BLOOM_CASE_01 = """

enum DecoderModelTensorIdx : uint32_t {
    // define inTensor
    // idx: 0, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_INPUT_IDS = 0,
    // idx: 1, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_POSITION_IDS,
    // idx: 2, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_COS_TABLE,
    // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_SIN_TABLE,
    // idx: 4, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    // PA: [maxInputLength, maxInputLength]
    IN_TENSOR_ATTENTION_MASK,
    // idx: 5, shape: [4, 9]; PA所需入参
    IN_TENSOR_BLOCK_TABLES,
    // idx: 6, shape: [seqLen]; PA所需入参
    IN_TENSOR_SLOTS,
    // idx: 7, shape: [1]; FA所需入参
    IN_TENSOR_LAYER_IDX,
    // idx: 8, shape: [batchSize]; FA所需入参
    IN_TENSOR_TOKEN_OFFSET,
    // idx: 9, shape: [1]
    IN_TENSOR_PLACE_HOLDER,
    // idx: 10, shape: FA: [batchSize] PA: [4]
    IN_TENSOR_SEQ_LEN,
    // idx: 11, shape: FA: [batchSize]  PA: [4]
    IN_TENSOR_LOGTIS_INDICES,
};

enum DecoderModelInternalTensorIdx : uint32_t {
    INTERNAL_TENSOR_HIDDEN_STATES = 0,
    // INTERNAL_TENSOR_TO_LAYERS
};
"""

CHATGLM_CASE_01 = """

enum DecoderModelTensorIdx : uint32_t {
    // define inTensor
    // idx: 0, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_INPUT_IDS = 0,
    // idx: 1, shape: FA: [batchSize, seqLen] PA: [seqLen]
    IN_TENSOR_POSITION_IDS,
    // idx: 2, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_COS_TABLE,
    // idx: 3, shape: FA: [maxPositionEmbeddings, hiddenSizePerAttentionHead]
    // PA: [maxInputLength, hiddenSizePerAttentionHead]
    IN_TENSOR_SIN_TABLE,
    // idx: 4, shape: FA: [batchSize, maxPositionEmbeddings, maxPositionEmbeddings]
    // PA: [maxInputLength, maxInputLength]
    IN_TENSOR_ATTENTION_MASK,
    // idx: 5, shape: [4, 9]; PA所需入参
    IN_TENSOR_BLOCK_TABLES,
    // idx: 6, shape: [seqLen]; PA所需入参
    IN_TENSOR_SLOTS,
    // idx: 7, shape: [1]; FA所需入参
    IN_TENSOR_KV_CACHE_IDX,
    // idx: 8, shape: [batchSize]; FA所需入参
    IN_TENSOR_TOKEN_OFFSET,
    // idx: 9, shape: [1]
    IN_TENSOR_PLACE_HOLDER,
    // idx: 10, shape: FA: [batchSize] PA: [4]
    IN_TENSOR_SEQ_LEN,
    // idx: 11, shape: FA: [batchSize]  PA: [4]
    IN_TENSOR_LOGTIS_INDICES,
    // idx 12, shape: [batchsize] dtype: int32
    IN_TENSOR_Q_LEN,
};

enum DecoderModelInternalTensorIdx : uint32_t {
    // define internelTensor
    // idx: 0, shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
    INTERNAL_TENSOR_HIDDEN_STATES = 0,
    // idx: 1, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    INTERNAL_TENSOR_COS_EMB,
    // idx: 2, shape: [batchSize * seqLen, hiddenSizePerAttentionHead]
    INTERNAL_TENSOR_SIN_EMB,
};
"""

class TestParserReg(TestCase):
    def test_llama(self):
        self.assertEqual(
            ['input_ids', 'input_embedding', 'positional_ids', 'cosine_table', 'sine_table', 'attention_mask', 'block_tables', 'slots', 'kv_cache_idx', 'token_offset', 'place_holder', 'seq_len', 'logits_indices'],
            parse_input_names(LLAMA_CASE_01),
        )
    
    def test_bloom(self):
        self.assertEqual(
            ['IN_TENSOR_INPUT_IDS', 'IN_TENSOR_POSITION_IDS', 'IN_TENSOR_COS_TABLE', 'IN_TENSOR_SIN_TABLE', 'IN_TENSOR_ATTENTION_MASK', 'IN_TENSOR_BLOCK_TABLES', 'IN_TENSOR_SLOTS', 'IN_TENSOR_LAYER_IDX', 'IN_TENSOR_TOKEN_OFFSET', 'IN_TENSOR_PLACE_HOLDER', 'IN_TENSOR_SEQ_LEN', 'IN_TENSOR_LOGTIS_INDICES'],
            parse_input_names(BLOOM_CASE_01),
        )

    def test_chatglm(self):
        self.assertEqual(
            ['IN_TENSOR_INPUT_IDS', 'IN_TENSOR_POSITION_IDS', 'IN_TENSOR_COS_TABLE', 'IN_TENSOR_SIN_TABLE', 'IN_TENSOR_ATTENTION_MASK', 'IN_TENSOR_BLOCK_TABLES', 'IN_TENSOR_SLOTS', 'IN_TENSOR_KV_CACHE_IDX', 'IN_TENSOR_TOKEN_OFFSET', 'IN_TENSOR_PLACE_HOLDER', 'IN_TENSOR_SEQ_LEN', 'IN_TENSOR_LOGTIS_INDICES', 'IN_TENSOR_Q_LEN'],
            parse_input_names(CHATGLM_CASE_01),
        )
    
    def test_qwen(self):
        self.assertEqual(
            ['IN_TENSOR_INPUT', 'IN_TENSOR_POSITIONIDS', 'IN_TENSOR_COS', 'IN_TENSOR_SIN', 'IN_TENSOR_ATTENTIONMASK', 'IN_TENSOR_BLOCK_TABLES', 'IN_TENSOR_SLOTS', 'IN_TENSOR_KV_CACHE_IDX', 'IN_TENSOR_TOKEN_OFFSET', 'IN_TENSOR_SEQ_LENGTHS', 'IN_TENSOR_LOGTIS_INDICES', 'IN_PLACEHOLDER'],
            parse_input_names(QWEN_CASE_01),
        )
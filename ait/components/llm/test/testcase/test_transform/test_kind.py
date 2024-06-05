from unittest import TestCase

from torch.nn import Linear, Embedding, LayerNorm, SiLU, GELU, Module

from ait_llm.transform.model_parser.kind import convert, attention, mlp, activation


class RotaryEmbedding(Module):
    def __init__(
            self,
            dim: int,
            base: int = 10000,
            max_position_embeddings: int = 2048,
            max_seq_len_cached: int = 2048
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_seq_len_cached


class RMSNorm(Module):
    def __init__(self, eps: float):
        super().__init__()
        self.variance_epsilon = eps


class Attention(Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Linear(16, 16, bias=True)
        self.k_proj = Linear(16, 16, bias=True)
        self.v_proj = Linear(16, 16, bias=True)
        self.o_proj = Linear(16, 16, bias=False)

        self.rotary_emb = RotaryEmbedding(
            dim=16,
            base=100,
            max_position_embeddings=100,
            max_seq_len_cached=100
        )


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = Linear(16, 16, bias=False)
        self.up_proj = Linear(16, 16, bias=False)
        self.down_proj = Linear(16, 16, bias=True)
        self.act_fn = SiLU()


class TestKind(TestCase):
    def test_linear(self):
        linear = Linear(16, 48, False)
        self.assertEqual(
            {
                "kind": "Linear",
                "in_features": 16,
                "out_features": 48,
                "bias": False
            },
            convert(linear)
        )

    def test_embedding(self):
        embedding1 = Embedding(16, 16, 4)
        embedding2 = Embedding(16, 48)
        self.assertEqual(
            {
                "kind": "Embedding",
                "num_embeddings": 16,
                "embedding_dim": 16,
                "padding_idx": 4
            },
            convert(embedding1)
        )
        self.assertEqual(
            {
                "kind": "Embedding",
                "num_embeddings": 16,
                "embedding_dim": 48,
                "padding_idx": None
            },
            convert(embedding2)
        )

    def test_attention(self):
        attn = Attention()
        sub = list(attn.children())

        self.assertEqual(
            {
                "structure": "q-k-v-o-r",
                "q": {
                    "kind": "Linear",
                    "in_features": 16,
                    "out_features": 16,
                    "bias": True
                },
                "k": {
                    "kind": "Linear",
                    "in_features": 16,
                    "out_features": 16,
                    "bias": True
                },
                "v": {
                    "kind": "Linear",
                    "in_features": 16,
                    "out_features": 16,
                    "bias": True
                },
                "o": {
                    "kind": "Linear",
                    "in_features": 16,
                    "out_features": 16,
                    "bias": False
                },
                "rope": {
                    "kind": "RotaryEmbedding",
                    "base": 100,
                    "dim": 16,
                    "max_position_embeddings": 100,
                    "max_seq_len_cached": 100
                }
            },
            attention(sub, 5)
        )

    def test_mlp(self):
        my_mlp = MLP()
        sub = list(my_mlp.children())
        self.assertEqual(
            {
                "ff": [
                    {
                        "kind": "Linear",
                        "in_features": 16,
                        "out_features": 16,
                        "bias": False
                    }, {
                        "kind": "Linear",
                        "in_features": 16,
                        "out_features": 16,
                        "bias": False
                    }, {
                        "kind": "Linear",
                        "in_features": 16,
                        "out_features": 16,
                        "bias": True
                    }
                ],
                "act": {"kind": "SiLU"}
            },
            mlp(sub),
        )

    def test_layernorm(self):
        layernorm1 = LayerNorm(12)
        layernorm2 = LayerNorm(36, 0.2, False, False)

        self.assertEqual(
            {
                "kind": "LayerNorm",
                "normalized_shape": 12,
                "eps": 1e-5,
                "element_affine": True,
                "bias": True
            },
            convert(layernorm1)
        )
        self.assertEqual(
            {
                "kind": "LayerNorm",
                "normalized_shape": 36,
                "eps": 0.2,
                "element_affine": False,
                "bias": False
            },
            convert(layernorm2)
        )

    def test_rope(self):
        rope1 = RotaryEmbedding(dim=12)
        rope2 = RotaryEmbedding(dim=36, base=100, max_position_embeddings=100)

        self.assertEqual(
            {
                "kind": "RotaryEmbedding",
                "base": 10000,
                "dim": 12,
                "max_position_embeddings": 2048,
                "max_seq_len_cached": 2048
            },
            convert(rope1)
        )
        self.assertEqual(
            {
                "kind": "RotaryEmbedding",
                "base": 100,
                "dim": 36,
                "max_position_embeddings": 100,
                "max_seq_len_cached": 2048
            },
            convert(rope2)
        )

    def test_rms_norm(self):
        rms_norm1 = RMSNorm(eps=0.2)
        rms_norm2 = RMSNorm(eps=1e-5)

        self.assertEqual(
            {
                "kind": "RMSNorm",
                "eps": 0.2
            },
            convert(rms_norm1)
        )
        self.assertEqual(
            {
                "kind": "RMSNorm",
                "eps": 1e-5
            },
            convert(rms_norm2)
        )

    def test_activation(self):
        silu = SiLU()
        gelu1 = GELU()
        gelu2 = GELU(approximate="tanh")

        self.assertEqual({"kind": "SiLU"}, activation(silu))
        self.assertEqual({"kind": "GELU", "approximate": False}, activation(gelu1))
        self.assertEqual({"kind": "GELU", "approximate": True}, activation(gelu2))

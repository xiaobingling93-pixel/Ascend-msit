from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import ForwardFactory

# 全局变量用于保存第一次获取的函数
FLUX_APPLY_ROTARY_EMB_MINDSPEED = None
FLUX_APPLY_FA = None
FLUX_ATTENTION = None

# 注册 Flux 模型的适配器
@ForwardFactory.register("FluxTransformerBlock", "FluxAttnProcessor2_0")
def flux_attn_processor_adapter(original_call):
    """FluxAttnProcessor2_0 的量化适配器"""
    global FLUX_APPLY_ROTARY_EMB_MINDSPEED, FLUX_APPLY_FA, FLUX_ATTENTION

    from importlib import import_module
    import torch.distributed as dist

    # 如果已经初始化过，直接使用保存的函数
    if all([FLUX_APPLY_ROTARY_EMB_MINDSPEED, FLUX_APPLY_FA, FLUX_ATTENTION]):
        apply_rotary_emb_mindspeed = FLUX_APPLY_ROTARY_EMB_MINDSPEED
        apply_fa = FLUX_APPLY_FA
        Attention = FLUX_ATTENTION
    else:
        # 第一次调用，从模块中获取并保存
        flux_attn_processor_module = import_module(original_call.__module__)
        
        apply_rotary_emb_mindspeed = flux_attn_processor_module.apply_rotary_emb_mindspeed
        apply_fa = flux_attn_processor_module.apply_fa
        Attention = flux_attn_processor_module.Attention
        
        # 保存到全局变量
        FLUX_APPLY_ROTARY_EMB_MINDSPEED = apply_rotary_emb_mindspeed
        FLUX_APPLY_FA = apply_fa
        FLUX_ATTENTION = Attention
    
    def new_call(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        if attn.is_tp:
            attn_heads = attn.heads // attn.world_size
        else:
            attn_heads = attn.heads
        head_dim = inner_dim // attn_heads

        query = query.view(batch_size, -1, attn_heads, head_dim)
        key = key.view(batch_size, -1, attn_heads, head_dim)
        value = value.view(batch_size, -1, attn_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn_heads, head_dim
        )
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn_heads, head_dim
        )
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn_heads, head_dim
        )

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=1)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb_mindspeed(query, image_rotary_emb)
            key = apply_rotary_emb_mindspeed(key, image_rotary_emb)

        # --------------------fa3-----------------------------
        query = attn.fa_quantizer.quant(query, qkv="q")
        key = attn.fa_quantizer.quant(key, qkv="k")
        value = attn.fa_quantizer.quant(value, qkv="v")
        # --------------------fa3-----------------------------
        
        hidden_states = apply_fa(query, key, value, attention_mask)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1]:],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if attn.is_tp:
            dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        if attn.is_tp:
            dist.all_reduce(encoder_hidden_states, op=dist.ReduceOp.SUM)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states
    return new_call


# 全局变量用于保存第一次获取的函数
FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED = None
FLUX_SINGLE_APPLY_FA = None
FLUX_SINGLE_ATTENTION = None


# 注册 Flux 模型的适配器
@ForwardFactory.register("FluxSingleTransformerBlock", "FluxSingleAttnProcessor2_0")
def flux_attn_single_processor_adapter(original_call):
    """FluxSingleAttnProcessor2_0 的量化适配器"""
    global FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED, FLUX_SINGLE_APPLY_FA, FLUX_SINGLE_ATTENTION

    from importlib import import_module
    import torch.distributed as dist

    # 如果已经初始化过，直接使用保存的函数
    if all([FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED, FLUX_SINGLE_APPLY_FA, FLUX_SINGLE_ATTENTION]):
        apply_rotary_emb_mindspeed = FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED
        apply_fa = FLUX_SINGLE_APPLY_FA
        Attention = FLUX_SINGLE_ATTENTION
    else:
        # 第一次调用，从模块中获取并保存
        flux_attn_processor_module = import_module(original_call.__module__)
        
        apply_rotary_emb_mindspeed = flux_attn_processor_module.apply_rotary_emb_mindspeed
        apply_fa = flux_attn_processor_module.apply_fa
        Attention = flux_attn_processor_module.Attention
        
        # 保存到全局变量
        FLUX_SINGLE_APPLY_ROTARY_EMB_MINDSPEED = apply_rotary_emb_mindspeed
        FLUX_SINGLE_APPLY_FA = apply_fa
        FLUX_SINGLE_ATTENTION = Attention


    def new_call(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        if attn.is_tp:
            attn_heads = attn.heads // attn.world_size
        else:
            attn_heads = attn.heads
        head_dim = inner_dim // attn_heads

        query = query.view(batch_size, -1, attn_heads, head_dim)

        key = key.view(batch_size, -1, attn_heads, head_dim)
        value = value.view(batch_size, -1, attn_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb_mindspeed(query, image_rotary_emb)
            key = apply_rotary_emb_mindspeed(key, image_rotary_emb)

        # --------------------fa3-----------------------------
        query = attn.fa_quantizer.quant(query, qkv="q")
        key = attn.fa_quantizer.quant(key, qkv="k")
        value = attn.fa_quantizer.quant(value, qkv="v")
        # --------------------fa3-----------------------------

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = apply_fa(query, key, value, attention_mask)
        hidden_states = hidden_states.to(query.dtype)
        B, S, H = hidden_states.shape
        if attn.is_tp:
            hidden_states_full = torch.empty(
                [attn.world_size, B, S, H], dtype=hidden_states.dtype, device=hidden_states.device
                )
            dist.all_gather_into_tensor(hidden_states_full, hidden_states)
            hidden_states = hidden_states_full.permute(1, 2, 0, 3).reshape([B, S, 2 * H])

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states
    return new_call


# 注册 HYVideo 模型的适配器
@ForwardFactory.register("MMDoubleStreamBlock", "double_forward")
def hyvideo_mm_double_stream_block_double_forward_adapter(original_forward):
    """HYVideo 模型的double_forward适配器"""
    from importlib import import_module
        
    hyvideo_double_module = import_module(original_forward.__module__)
    modulate = hyvideo_double_module.modulate
    rearrange = hyvideo_double_module.rearrange
    apply_rotary_emb = hyvideo_double_module.apply_rotary_emb
    attention = hyvideo_double_module.attention
    parallel_attention = hyvideo_double_module.parallel_attention

    def new_double_forward(
            self,
            img, txt,
            img_mod1_shift,
            img_mod1_scale,
            txt_mod1_shift,
            txt_mod1_scale,
            freqs_cis,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv
        ):
        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        
        # --------------------fa3-----------------------------
        q = self.fa_quantizer.quant(q, qkv="q")
        k = self.fa_quantizer.quant(k, qkv="k")
        v = self.fa_quantizer.quant(v, qkv="v")
        # --------------------fa3-----------------------------

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                mode="torch",
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                scale=self.scale
            )
        
        return attn
    return new_double_forward


@ForwardFactory.register("MMSingleStreamBlock", "single_forward")
def hyvideo_mm_single_stream_block_single_forward_adapter(original_forward):
    """HYVideo 模型的 single_forward 适配器"""
    from importlib import import_module
    
    hyvideo_single_module = import_module(original_forward.__module__)
    rearrange = hyvideo_single_module.rearrange
    apply_rotary_emb = hyvideo_single_module.apply_rotary_emb
    attention = hyvideo_single_module.attention
    parallel_attention = hyvideo_single_module.parallel_attention

    def new_single_forward(
            self,
            qkv,
            freqs_cis,
            txt_len,
            x,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv
        ):
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        
        # --------------------fa3-----------------------------
        q = self.fa_quantizer.quant(q, qkv="q")
        k = self.fa_quantizer.quant(k, qkv="k")
        v = self.fa_quantizer.quant(v, qkv="v")
        # --------------------fa3----------------------------- 

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                mode="torch",
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                scale=self.scale
            )
        # attention computation end
        return attn       
    return new_single_forward

@ForwardFactory.register("deepseekv3", "mla")
@ForwardFactory.register("deepseek_v3", "mla")
@ForwardFactory.register("deepseekv2", "mla")
@ForwardFactory.register("deepseek_v2", "mla")
def deepseekv2_mla_forward_adapter(original_forward):
    """DeepSeek V2/V3模型的MLA forward适配器"""

    from importlib import import_module
    from transformers import Cache
    from torch import nn
    from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import PrepareWeight

    deepseek_module = import_module(original_forward.__module__)
    apply_rotary_pos_emb = deepseek_module.apply_rotary_pos_emb

    def new_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37.\n"
                "Please make sure to use `attention_mask` instead."
            )
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv_seq_len = k_pe.shape[-2]

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(q_pe, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            compressed_kv = compressed_kv.unsqueeze(1)
            k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
            compressed_kv = compressed_kv.squeeze(1)

        with PrepareWeight(self.kv_b_proj):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)

        q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]
        out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]

        q_nope = torch.matmul(q_nope, q_absorb)

        # ----------FA3-------------
        q_nope = self.fa_quantizer.quant(q_nope, qkv="q")
        compressed_kv = self.fa_quantizer.quant(compressed_kv.unsqueeze(1), qkv="k").squeeze(1)
        _ = self.fa_quantizer.quant(compressed_kv.unsqueeze(1), qkv="v").squeeze(1)
        # ----------FA3-------------

        attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT))
        attn_weights = attn_weights * self.softmax_scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is None:
            raise ValueError("Attention mask cannot be None")
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_pe.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)
        attn_output = torch.matmul(attn_output, out_absorb.mT)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return new_forward
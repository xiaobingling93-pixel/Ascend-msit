# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import math
import random
import os
import json
import stat

from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DynamicCache
from ascend_utils.common.security import check_type, get_valid_write_path

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.omni_attention_pattern.omni_config import OmniAttentionConfig
from msmodelslim.pytorch.omni_attention_pattern.omni_utils import patch_with_omni_attn_pattern

# 用于评分question的数量
ORDINAL_NUMBERS = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth"
]

EPSILON = 5e-4


class OmniAttentionGeneticSearcher:
    def __init__(self, config: OmniAttentionConfig):
        """
        初始化 OmniAttentionGeneticSearcher 类。

        参数:
        config (OmniAttentionConfig): 包含模型配置和搜索参数的对象。
        """
        if not isinstance(config, OmniAttentionConfig):
            raise TypeError("config should be OmniAttentionConfig")

        self.config = config
        self.sparsity = 90

        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.out_dir = os.path.join(config.save_path, self.config._model_name)
        os.makedirs(self.out_dir, exist_ok=True)

        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model_config = AutoConfig.from_pretrained(config.model_path)

        self.num_layers = model_config.num_hidden_layers
        self.num_kv_heads = model_config.num_key_value_heads

        msmodelslim_logger.info(f"Loading model and tokenizer from path {config.model_path}.")
        self.load_model_and_tokenizer()
        self.tokenize_inputs()

    @property
    def num_ones(self):
        return int(self.num_layers * (1 - self.sparsity / 100))

    def load_model_and_tokenizer(self):
        """
        从指定的模型路径加载模型和分词器。
        """
        path = self.config.model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.device = self.model.device

    def tokenize_inputs(self):
        """
        对输入数据进行分词处理。
        """
        with open(os.path.join(self.work_dir, 'data.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.num_books = len(data)
        self.prompts: list[torch.LongTensor] = []
        self.answers: list[list[str]] = []

        for i in range(self.num_books):
            tokenized = self.tokenizer(data[i]['text'], add_special_tokens=False, return_tensors='pt').input_ids
            self.prompts.append(tokenized.to(self.device))
            self.answers.append(data[i]['answer'])


    def generate_gene_pool(self) -> list[np.ndarray]:
        """
        生成一组随机的二进制矩阵（基因池）。

        每个矩阵的形状为`(num_layers, num_kv_heads)`，其中1的比例由当前阶段（`stage`）决定。
        例如，如果`stage`=1，则1的比例为10%（稀疏度为90%）。
        总共生成`pool_size`个矩阵。
        生成的随机矩阵确保每一行的元素要么全为0，要么全为1。

        返回:
        list[np.ndarray]: 生成的二进制矩阵列表。
        """
        pool = []
        num_ones = self.num_ones

        for _ in range(self.config.pool_size):
            mat = np.zeros((self.num_layers, 1), dtype=int)
            mat[np.random.permutation(self.num_layers)[:num_ones], 0] = 1

            # 将矩阵沿列方向重复`num_kv_heads`次，扩展为`(num_layers, num_kv_heads)`形状
            mat = mat.repeat(self.num_kv_heads, axis=1)

            if all((mat != m).any() for m in pool):
                pool.append(mat)

        return pool

    @torch.no_grad()
    def score_one(self, pattern: np.ndarray) -> float:
        """
        对给定的注意力模式进行评分。

        参数:
        pattern (np.ndarray): 注意力模式矩阵，形状为 `(num_layers, num_kv_heads)`。

        返回:
        float: 当前模式的得分。
        """

        # 初始化得分
        score = 0

        # 使用给定的注意力模式修补模型
        patch_with_omni_attn_pattern(
            self.model,
            pattern
        )

        # 初始化进度条，显示评分进度
        pbar = tqdm(
            total=self.num_books * len(ORDINAL_NUMBERS),
            desc="Scoring current pattern...",
            leave=False,
            position=1)

        for book_id in range(self.num_books):
            prompt, answer = self.prompts[book_id], self.answers[book_id]
            for question_id, num in enumerate(ORDINAL_NUMBERS):
                question = (
                    f"\nBased on the content of the book, what is the {num} passkey to the vault?\n"
                    f"Answer: The {num} passkey is:\n")
                tokenized = self.tokenizer(
                    question,
                    add_special_tokens=False,
                    return_tensors='pt').input_ids.to(self.device)
                query = torch.cat([prompt, tokenized], dim=-1)

                # 初始化动态缓存
                cache = DynamicCache()
                out = self.model(query, use_cache=True, past_key_values=cache)

                generated_token = out.logits[:, -1, :].argmax(-1).unsqueeze(0)
                response = [generated_token.item()]
                cache = out.past_key_values

                # 继续生成后续的 token（总共生成 4 个 token）
                for _ in range(3):
                    out = self.model(generated_token, use_cache=True, past_key_values=cache)
                    generated_token = out.logits[:, -1, :].argmax(-1).unsqueeze(0)
                    response.append(generated_token.item())
                    cache = out.past_key_values

                response = self.tokenizer.decode(response)

                # 如果生成的文本中包含正确答案，则增加得分
                if answer[question_id] in response:
                    score += 1

                pbar.update(1)

        pbar.close()
        return score

    def search_incremental(self):
        """
        执行遗传搜索算法，寻找最佳的注意力模式。

        该方法通过多个阶段（stage）逐步优化注意力模式，每个阶段生成一组基因池并对其进行评分，
        最终选择最佳的模式并保存结果。
        """
        # 初始化每个注意力头的得分矩阵和出现次数矩阵
        score_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float) + EPSILON
        occur_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float) + EPSILON
        prev_best = None

        score_array_this_stage = None

        for stage in range(1, 10):
            # 设置当前稀疏度，从90到10
            self.sparsity = 100 - stage * 10
            best_score, best = -100, None

            if stage == 1:
                pool = self.generate_gene_pool()
            else:
                pool = self.evolution(score_array_this_stage, prev_best)

            for pattern in tqdm(pool, position=0, desc=f"Current search stage: {stage}", leave=False):
                score = self.score_one(pattern)

                score_per_head += pattern * score
                occur_per_head += pattern

                # 如果当前模式的得分更高，则更新最佳得分和最佳模式
                if score > best_score:
                    best_score = score
                    best = pattern

            score_array_this_stage = score_per_head / occur_per_head

            # 对当前阶段的得分数组进行变异操作，生成新的模式
            mutations = self.mutation(score_array_this_stage)

            # 对变异后的模式进行评分
            for pattern in tqdm(mutations, position=0, desc=f"Current search stage: {stage}. In mutation", leave=False):
                score = self.score_one(pattern)

                score_per_head += pattern * score
                occur_per_head += pattern

                if score > best_score:
                    best_score = score
                    best = pattern

            out_file = os.path.join(
                self.out_dir,
                f'genetic_rowwise_sparsity_{self.sparsity:d}_score_{best_score}.tsv')
            out_file = get_valid_write_path(out_file, warn_exists=True)
            np.savetxt(out_file, 1 - best, delimiter='\t', fmt='%d')
            os.chmod(out_file, stat.S_IRUSR)
            msmodelslim_logger.info(f"Saving best pattern to path {out_file}.")

            prev_best = best.copy()

    def search_on_this_sparsity(self, sparsity):
        """
        执行遗传搜索算法，寻找最佳的注意力模式。
        遗传算法通过进化生成模式池，对每个模式进行评分，并逐轮优化，直到得分不再提升为止。
        """
        if sparsity is None:
            raise ValueError("Please check sparsity, should be an integer greater than zero and less than 100")
        check_type(sparsity, int, param_name="sparsity")
        if sparsity <= 0 or sparsity >= 100:
            raise ValueError("Please check sparsity, should be an integer greater than zero and less than 100")

        self.sparsity = sparsity
        score_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float) + EPSILON
        occur_per_head = np.zeros((self.num_layers, self.num_kv_heads), dtype=float) + EPSILON
        score_array_this_stage = np.ones((self.num_layers, self.num_kv_heads), dtype=float)

        best_score_this_round = -10  # 当前轮次的最佳得分
        best_score_last_round = -100  # 上一轮次的最佳得分
        genetic_round = 0  # 遗传算法的轮数

        historical_patterns_lis = []    # 用于记录之前已经搜索过的模式，避免重复计算
        best_pattern = np.zeros((self.num_layers, self.num_kv_heads), dtype=float)

        # 用于判断是否继续进化，如果本轮最佳得分不超过上一轮最佳得分，则停止搜索
        while best_score_this_round > best_score_last_round:
            best_score_last_round = best_score_this_round

            pool = self.evolution(
                score_array_this_stage,
                best_pattern,
                a_with_random=True,
                historical_patterns_lis=historical_patterns_lis
            )

            for pattern in tqdm(pool, position=0, desc=f"Current round: {genetic_round}", leave=False):
                score = self.score_one(pattern)
                historical_patterns_lis.append(pattern)

                score_per_head += pattern * score
                occur_per_head += pattern

                if score > best_score_this_round:
                    best_score_this_round = score
                    best_pattern = pattern

            score_array_this_stage = score_per_head / occur_per_head
            genetic_round += 1

        out_file = os.path.join(
            self.out_dir,
            f'genetic_rowwise_on_this_sparsity_{self.sparsity:d}_score_{best_score_this_round}.tsv'
        )
        out_file = get_valid_write_path(out_file, warn_exists=True)
        # 将最佳模式保存为文件，格式为 TSV，每个值为 0 或 1（1 表示稀疏位置）
        np.savetxt(out_file, 1 - best_pattern, delimiter='\t', fmt='%d')
        os.chmod(out_file, stat.S_IRUSR)
        msmodelslim_logger.info(f"Saving best pattern with sparsity {self.sparsity:d} to path {out_file}.")

    def mutation(self, score_per_head: np.ndarray) -> list[np.ndarray]:
        """
        对基因进行变异操作，生成一组新的基因, 用于基于得分的小型排序。

        参数:
        score_per_head (np.ndarray): 每个注意力头的得分矩阵，形状为 `(num_layers, num_kv_heads)`。

        返回:
        list[np.ndarray]: 变异后的基因矩阵列表，每个矩阵的形状为 `(num_layers, num_kv_heads)`。
        """
        num_ones = self.num_ones
        score_per_layer = score_per_head.sum(-1)
        rank_per_layer = score_per_layer.argsort().argsort()
        best_genes_mask = (rank_per_layer >= self.num_layers - num_ones).astype(int)

        ones_pos = np.nonzero(best_genes_mask)[0]
        zeros_pos = np.nonzero(1 - best_genes_mask)[0]
        mutations = [best_genes_mask]

        # 生成指定数量的变异基因
        for _ in range(self.config.num_mutation):
            # 随机选择一个被选中的层和一个未被选中的层
            a = np.random.choice(ones_pos, size=1)
            b = np.random.choice(zeros_pos, size=1)

            # 复制最佳基因掩码并进行变异操作
            mutated = best_genes_mask.copy()
            mutated[a] = 1 - mutated[a]  # 将被选中的层从1变为0
            mutated[b] = 1 - mutated[b]  # 将未被选中的层从0变为1

            # 确保变异后的基因中1的数量与预期一致
            if mutated.sum() != num_ones:
                raise RuntimeError(
                    f"The mutation {mutated} has wrong number of ones. "
                    f"Should be {num_ones}, but got {mutated.sum()} instead.")

            # 将变异后的基因添加到结果列表中
            mutations.append(mutated)

        # 将变异后的基因掩码扩展为 `(num_layers, num_kv_heads)` 形状
        for i, mutation in enumerate(mutations):
            mutations[i] = mutation[:, None].repeat(self.num_kv_heads, axis=1)
        return mutations

    def evolution(
            self,
            score_per_head: np.ndarray,
            best_pattern: np.ndarray,
            pool_size: int = -1,
            ab_rate: float = 0.8,
            prob_bias_rate: float = 144,
            a_with_random: bool = False,
            historical_patterns_lis: list[np.ndarray] = None,
    ) -> list[np.ndarray]:
        """
        生成新的模式池,支持检查是否与历史模式重复的版本。
        目前仅支持进化模式和同等stage里的继续进化, 以及退化(从更高级的stage退化)

        参数:
        - score_per_head: 每个头的得分,形状为 (dim1, dim2)。
        - best_pattern: 最佳模式,形状为 (dim1, dim2)。可以使用历史stage或当前stage的best_pattern。
            当使用当前stage的best_pattern来优化当前搜索成果时,建议将a_with_random设置为True以保留更多遗传信息。
        - pool_size: 需要搜索的pattern数量，不设置的话会使用 self.config.pool_size。
        - ab_rate: A部分的比例,默认为0.8。A部分更可能包含best_pattern,B部分更随机。
            当a_with_random为False时,A部分必包含best_pattern；当a_with_random为True时,A部分会赋予更高的稳定性。
        - prob_bias_rate: 概率偏置率,默认为6.0。
        - a_with_random: 是否在A部分添加随机性, 进化生成更多的1的时候默认为False, 其他情况建议设置为True。
        - historical_patterns_lis: 历史已搜索的模式列表,默认为None。

        返回:
        - 新的模式池,包含生成的模式。
        """
        total1s = self.num_ones
        dim1 = self.num_layers  # 层的数量
        dim2 = self.num_kv_heads  # 每个层的kv_heads数量
        dim22 = 1  # 用于reshape的辅助维度

        if historical_patterns_lis is None:
            historical_patterns_lis = []

        ordermx = score_per_head.sum(-1).reshape(dim1, 1)
        best_pattern_full_row = (np.sum(best_pattern.copy(), 1) / dim2 + 0.5).astype(int).reshape((dim1, dim22))
        keep = int(np.sum(best_pattern.reshape(-1)) / dim2 + 0.5)

        # 计算A和B部分的数量,   pool_size is the expected amount of gene pools.
        if pool_size < 0 :
            pool_size = self.config.pool_size

        amount_a = int(pool_size * ab_rate + 0.1)
        amount_b = int(pool_size - amount_a + 0.1)

        # 调整amountA和amountB,确保不超过组合数的限制
        amount_his = len(historical_patterns_lis)
        best_pattern_full_row_1s = np.sum(best_pattern_full_row)
        his_a = 0

        for i in range(amount_his):
            pattern_full_row = np.sum(best_pattern_full_row.reshape(-1) *
                                      (np.sum(historical_patterns_lis[i], 1) / dim2 + 0.5).astype(int).reshape(-1))
            if pattern_full_row == best_pattern_full_row_1s:
                his_a += 1

        if a_with_random:
            amount_a = min(math.comb(dim1 * dim22, total1s) - amount_his, amount_a)
        else:
            amount_a = min(math.comb(dim1 * dim22 - keep, total1s - keep) - his_a, amount_a)
        amount_b = min(math.comb(dim1 * dim22, total1s) - amount_a - amount_his, amount_b)

        # 初始化原型矩阵,用于生成新的模式
        prototype_matrix = np.ones((dim1, dim2))

        # 计算A部分的采样概率
        min_a = np.min(ordermx[best_pattern_full_row < 0.5])
        max_a = np.max(ordermx[best_pattern_full_row < 0.5])

        if min_a != max_a:
            prob_a = ((1. - best_pattern_full_row) * ordermx - min_a) / (max_a - min_a) * (prob_bias_rate - 1) + 1.
        else:
            prob_a = np.ones((dim1, dim22))
        prob_a = prob_a * (1. - best_pattern_full_row)
        prob_a = prob_a / np.sum(prob_a)

        # 初始化索引数组
        indexes = np.array(range(dim1 * dim22))
        new_a = int(total1s - keep + 0.5)

        # 如果A_with_random为True,则调整probA并增加随机性
        if a_with_random:
            prob_a += best_pattern_full_row * np.sum(prob_a) * 9.
            prob_a = prob_a / np.sum(prob_a)
            new_a = total1s

        # 初始化输出列表
        output = []
        count_a = 0

        # 生成A部分的模式
        while count_a < amount_a:
            if a_with_random:
                matrix_a_new = np.zeros((dim1, dim22))
            else:
                matrix_a_new = best_pattern_full_row.copy()

            # 随机采样A部分的模式
            sample_a = np.random.choice(indexes, new_a, replace=False, p=prob_a.reshape(-1))

            for index in sample_a:
                matrix_a_new[index // dim22, index % dim22] = 1.

            # 检查生成的模式是否符合要求
            if int(np.sum(matrix_a_new) + 0.5) != total1s:
                msmodelslim_logger.warning("The newly generated quantity doesn't match the total quantity")
                continue

            matrix_a_new = matrix_a_new * prototype_matrix

            # 检查生成的模式是否已经存在于输出列表中
            if (not any((matrix_a_new == arr).all() for arr in output) and
                    not any((matrix_a_new == arr).all() for arr in historical_patterns_lis)):
                output.append(matrix_a_new)
                count_a += 1

        # 计算B部分的采样概率
        min_b = np.min(ordermx.reshape(-1))
        max_b = np.max(ordermx.reshape(-1))

        if min_b != max_b:
            prob_b = (ordermx - min_b) / (max_b - min_b) * (prob_bias_rate - 1) + 1.
        else:
            prob_b = np.ones((dim1, dim22))
        prob_b = prob_b / np.sum(prob_b)

        new_b = total1s
        count_b = 0

        # 生成B部分的模式
        while count_b < amount_b:
            matrix_b_new = np.zeros((dim1, dim22))

            # 随机采样B部分的模式
            sample_b = np.random.choice(indexes, new_b, replace=False, p=prob_b.reshape(-1))

            for index in sample_b:
                matrix_b_new[index // dim22, index % dim22] = 1.

            # 检查生成的模式是否符合要求
            if int(np.sum(matrix_b_new) + 0.5) != total1s:
                msmodelslim_logger.warning("The newly generated quantity doesn't match the total quantity")
                continue

            matrix_b_new = matrix_b_new * prototype_matrix

            # 检查生成的模式是否已经存在于输出列表中
            if (not any((matrix_b_new == arr).all() for arr in output) and
                    not any((matrix_b_new == arr).all() for arr in historical_patterns_lis)):
                output.append(matrix_b_new)
                count_b += 1

        # 检查输出数量是否匹配
        if len(output) != amount_a + amount_b:
            msmodelslim_logger.warning("The output quantity doesn't match the total quantity")

        return output
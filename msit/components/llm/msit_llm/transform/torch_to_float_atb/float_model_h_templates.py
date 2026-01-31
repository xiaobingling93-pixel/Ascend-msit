# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
COPYRIGHT_HEADER = """/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          {licenses_url}
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */
"""

INCLUDE_HEADER_FORMATTER = """
#ifndef ATB_SPEED_MODELS_{model_name_upper}_DECODER_MODEL_H
#define ATB_SPEED_MODELS_{model_name_upper}_DECODER_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
"""

BASIC_CLASS_FORMATTER = """
namespace atb_speed {{
namespace {model_name_lower} {{
class DecoderModel : public Model {{
public:
    {STRUCT_PARAM_FORMATTER}

    explicit DecoderModel(const std::string &param);
    ~DecoderModel();
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    Param param_;
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
}};

REGISTER_MODEL({model_name_lower}, DecoderModel);

}}  // namespace {model_name_lower}
}}  // namespace atb_speed
#endif
"""

STRUCT_PARAM_FORMATTER = """
    struct Param {{
        // isFA为true则使用Flash Attention; 反之，则使用Paged Attention
        bool isFA = false;
        // isPrefill为true时为全量阶段，encoder的isPrefill参数应为true; isPrefill为false时为增量阶段，decoder的isPrefill参数应为false
        bool isPrefill = false;
        // isBF16为true时采用BF16精度; 反之，则采用FP16精度
        bool isBF16 = false;
        // withEmbedding为true时，模型包含word embedding层; 反之输入为hidden states; 该选项用于多模态模型适配
        bool withEmbedding = true;
        // isEmbeddingParallel为true时，embedding的权重在hiddenSize维度进行切分; 反之，则不对权重进行切分; 测试表明embedding切分并不会带来性能提升
        bool isEmbeddingParallel = false;
        // isLmHeadParallel为true时，LmHead的权重在vacobSize维度进行切分; 反之，则不对权重进行切分
        bool isLmHeadParallel = true;
        // 0 - No quant; 1- Quant in RmsNorm，dequant in Linear; 2 - Both quant and dequant in Linear
        int lmHeadTransposeType = -1;
        bool supportSwiGLU = false;
        // MLP是否使用SwiGLU，若为true时，则使用；反之，使用swish
        bool supportSpeculate = false;
        float rmsNormEps = 0;
        int numAttentionHeadsPerRank = 0;
        int hiddenSizePerAttentionHead = 0;
        int numHiddenLayers = 0;
        int numKeyValueHeadsPerRank = 0;
        bool supportLcoc = false;
        int rank = 0;
        int worldSize = 1;
        bool enableLogN = false;
        std::string backend = "hccl";
        std::vector<int> tokenOffset = {{}};
        std::vector<int> seqLen = {{}};
        std::vector<std::vector<int>> packQuantType = {{}};
        std::vector<std::vector<int>> linearQuantType = {{}};
        std::vector<std::vector<int>> linearTransposeType = {{}};
        void FromString(const std::string &param);
    }};
"""
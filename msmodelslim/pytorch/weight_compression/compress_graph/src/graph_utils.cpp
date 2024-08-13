/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: graph_utils.h for weight compression
 * Author: Huawei
 * Create: 2023-09-21
 */

#include "graph_utils.h"

using namespace std;

int CheckShape(vector<int64_t> &shape)
{
    MAXVAL = 50000
    for (auto const& val:shape) {
        if (val > MAXVAL) {
            return FAILED;
        }
    }
    return SUCCESS;
}

void GetDataSizeFromShape(vector<int64_t> shape, int64_t &size)
{
    for (const auto& val:shape) {
        size *= val;
    }
}

bool GetDataFromBin(string input_path, vector<int64_t> shapes, uint8_t** data, int dataTypeSize)
{
    ifstream inFile(input_path.c_str(), std::ios::in | std::ios::binary);
    if (!inFile.is_open()) {
        std::cout << "Failed to open" << input_path.c_str() << std::endl;
        return false;
    }
    inFile.seekg(0, ios_base::end);
    istream::pos_type fileSize = inFile.tellg();
    inFile.seekg(0, ios_base::beg);

    int64_t size = 1;
    GetDataSizeFromShape(shapes, size);
    uint32_t dataLen = size * dataTypeSize;
    if (dataLen != fileSize) {
        std::cout << "Invalid Param.len:" << dataLen << " is not equal with binary size." << fileSize << ")\n";
        inFile.close();
        return false;
    }
    char* pdata = new(std::nothrow) char[dataLen];
    if (pdata == nullptr) {
        std::cout << "Invalid pdata ptr.\n";
        inFile.close();
        return false;
    }
    inFile.read(reinterpret_cast<char*>(pdata), dataLen);
    *data = reinterpret_cast<uint8_t*>(pdata);
    if (inFile.fail()) {
        std::cout << "Read file failed.\n";
    }
    if (pdata == nullptr) {
        return false;
    }
    return true;
}

int32_t BuildCompressFcGraph(Graph &graph, uint8_t* data, vector<int64_t> &shape, vector<int64_t> &compressParameters)
{
    /*
        weight
          |
      CompressFc
      /        \
    compress  compress_index
    */
    auto shape_weight = shape;
    int64_t compress_size = 1;
    GetDataSizeFromShape(shape_weight, compress_size);

    TensorDesc desc_weight(ge::Shape(shape_weight), FORMAT_FRACTAL_NZ, DT_INT8);
    Tensor weight_tensor(desc_weight);
    weight_tensor.SetData(data, compress_size);
    auto weight = op::Const("weight_compress").set_attr_value(weight_tensor);

    auto dimK = shape_weight[0];
    auto dimN = shape_weight[1];
    auto tileNumN = (dimN + 8 - 1) / 8;
    auto tileNumK = (dimK + 8 - 1) / 8;
    int64_t index_size = 8 * tileNumN * tileNumK;
    std::cout<< "compress_size: "<< compress_size<<" index_size: "<<index_size<<std::endl;
    auto compressFc = op::CompressFcOp("compress_fc")
        .set_input_weight(weight)
        .set_attr_compress_parameters(compressParameters);
    compressFc.SetAttr("alg", "weight_unzip");

    TensorDesc desc_weight_compress(ge::Shape({compressParameters[0]}), FORMAT_ND, DT_INT8);

    auto shape_index = vector<int64_t>({8 * tileNumN * tileNumK});
    std::cout<<"tileNumN: "<<tileNumN<<" tileNumK: "<<tileNumK<<std::endl;
    TensorDesc desc_index(ge::Shape(shape_index), FORMAT_ND, DT_UINT8);
    TensorDesc desc_tiling_info(ge::Shape({3}), FORMAT_ND, DT_UINT32);

    compressFc.update_input_desc_weight(desc_weight);
    compressFc.update_output_desc_weight_compress(desc_weight_compress);
    compressFc.update_output_desc_compress_index(desc_index);
    compressFc.update_output_desc_compress_info(desc_tiling_info);

    vector<Operator> inputs = {weight};
    vector<Operator> outputs = {compressFc};

    graph.SetInputs(inputs).SetOutputs(outputs);

    return SUCCESS;
}

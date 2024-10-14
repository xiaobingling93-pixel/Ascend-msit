#include <cstring>
#include <memory>
#include <iostream>
#include <algorithm>
#include "ge/ge_ir_build.h"
#include "ge/ge_api_error_codes.h"

bool SaveOM(const void *model, size_t length, const std::string &file_path)
{
    if (length <= 0 || model == nullptr) {
        return false;
    }

    uint8_t* om_buffer = nullptr;
    try {
        om_buffer = new uint8_t[length];
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return false;
    }

    // 没有结束迭代器的话，使用copy_n更加规范
    std::copy_n(static_cast<const uint8_t*>(model), length, om_buffer);

    ge::ModelBufferData om_model;
    // std::shared_ptr默认使用delete来释放内存，会导致对new[]的ub行为。所以自定义为delete[]
    om_model.data = std::shared_ptr<uint8_t>(om_buffer, [](uint8_t* p) {delete[] p;});
    om_model.length = length;

    auto ret = ge::aclgrphSaveModel(file_path.c_str(), om_model);
    if (ret != ge::SUCCESS) {
        return false;
    }
    return true;
}
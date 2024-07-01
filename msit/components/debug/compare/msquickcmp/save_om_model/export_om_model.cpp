#include <cstring>
#include <memory>
#include <iostream>
#include "ge/ge_ir_build.h"
#include "ge/ge_api_error_codes.h"

bool SaveOM(const void *model, size_t length, const std::string &file_path)
{
    if (length <= 0) {
        return false;
    }
    auto om_buffer = new uint8_t[length];
    std::memcpy(om_buffer, model, length);
    ge::ModelBufferData om_model;
    om_model.data = std::shared_ptr<uint8_t>(om_buffer);
    om_model.length = length;
    auto ret = ge::aclgrphSaveModel(file_path.c_str(), om_model);
    if (ret != ge::SUCCESS) {
        return false;
    }
    return true;
}
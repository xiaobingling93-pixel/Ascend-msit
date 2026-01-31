/*
 * -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          `http://license.coscl.org.cn/MulanPSL2`  
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */
#ifndef ATB_SPEED_UTILS_OPERATION_FACTORY_H
#define ATB_SPEED_UTILS_OPERATION_FACTORY_H

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include "atb/operation.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
using CreateOperationFuncPtr = std::function<atb::Operation *(const nlohmann::json &)>;

class OperationFactory {
public:
    static bool Register(const std::string &operationName, CreateOperationFuncPtr createOperation);
    static atb::Operation *CreateOperation(const std::string &operationName, const nlohmann::json &param);
    static std::unordered_map<std::string, CreateOperationFuncPtr> &GetRegistryMap();
};
};
#endif

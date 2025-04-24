# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from msprechecker.prechecker.register import PrecheckerBase
from msprechecker.prechecker.utils import logger, get_npu_info, get_global_env_info
from msprechecker.prechecker.utils import parse_mindie_server_config, parse_ranktable_file
from msprechecker.prechecker.utils import get_model_path_from_mindie_config, get_mindie_server_config
from msprechecker.prechecker.utils import is_deepseek_model, read_csv_or_json
from msprechecker.prechecker.suggestions import GLOBAL_DEFAULT_CONFIG, DOMAIN, NOT_EMPTY_VALUE
from msprechecker.prechecker.suggestions import update_to_default_suggestions, suggestion_rule_checker


class ConfigCheckerBase(PrecheckerBase):
    __checker_name__ = "Config"

    def __init__(self, domain):
        super().__init__()
        self.domain, config_path = domain, ""

    def action(self, env_key, env_value):
        if env_value == NOT_EMPTY_VALUE:
            return f"配置文件 {self.config_path} 中添加 {env_key} 字段"
        else:
            return f"配置文件 {self.config_path} 中修改 {env_key}={env_value}"

    def do_precheck(self, current_config, additional_checks=None, **kwargs):
        if not current_config:
            return
        update_to_default_suggestions(self.domain, additional_checks)

        env_info = get_global_env_info()
        env_info["NPU_TYPE"] = get_npu_info(to_inner_type=True)  # Value like A2 A3
        current_config.update(env_info)  # Also update env values into config
        for suggestion_rule in GLOBAL_DEFAULT_CONFIG.get(self.domain, []):
            result, suggestion_value, current_value = suggestion_rule_checker(
                current_config, suggestion_rule, env_info, domain=self.domain, action_func=self.action
            )


class MindieConfigChecker(ConfigCheckerBase):
    __checker_name__ = "MindieConfig"

    def __init__(self):
        super().__init__(domain=DOMAIN.mindie_config)

    def collect_env(self, mindie_service_path=None, **kwargs):
        self.config_path = get_mindie_server_config(mindie_service_path)
        return parse_mindie_server_config(self.config_path)


class RankTableChecker(ConfigCheckerBase):
    __checker_name__ = "RankTable"

    def __init__(self):
        super().__init__(domain=DOMAIN.ranktable)

    def collect_env(self, ranktable_file=None, **kwargs):
        self.config_path = ranktable_file
        return parse_ranktable_file(self.config_path)


class ModelConfigChecker(ConfigCheckerBase):
    __checker_name__ = "ModelConfig"

    def __init__(self):
        super().__init__(domain=DOMAIN.model_config)

    def collect_env(self, mindie_service_path=None, **kwargs):
        model_name, model_weight_path = get_model_path_from_mindie_config(mindie_service_path=mindie_service_path)

        if not model_name or not model_weight_path:
            return None

        model_config, model_config_path = {}, os.path.join(model_weight_path, "config.json")
        if os.path.exists(model_config_path):
            model_config = read_csv_or_json(model_config_path)
        self.config_path = model_config_path
        logger.debug(f"ModelConfigCollecter model_name={model_name} model_config={model_config}")
        return {"model_name": model_name, "model_config": model_config}

    def do_precheck(self, current_config, additional_checks=None, **kwargs):
        if not current_config:
            return
        model_name, model_config = current_config.get("model_name", None), current_config.get("model_config", None)
        if not model_name or not model_config:
            return
        if not is_deepseek_model(model_name):
            return

        super().do_precheck(current_config=model_config, additional_checks=additional_checks, **kwargs)


class UserConfigChecker(ConfigCheckerBase):
    __checker_name__ = "UserConfig"

    def __init__(self):
        super().__init__(domain=DOMAIN.user_config)

    def collect_env(self, user_config_path=None, **kwargs):
        self.config_path = user_config_path
        return read_csv_or_json(user_config_path) if user_config_path and os.path.exists(user_config_path) else {}


mindie_config_checker = MindieConfigChecker()
ranktable_checker = RankTableChecker()
model_config_checker = ModelConfigChecker()
user_config_checker = UserConfigChecker()

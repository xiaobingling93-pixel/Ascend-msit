# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing_extensions import Self, Type


class ModelslimError(Exception):
    code = 0
    default_message = 'modelslim error'

    @classmethod
    def create_exception(cls, name: str, code: int, default_message: str = '') -> Type[Self]:
        return type(name, (cls,), {"code": code, "message": default_message})

    def __init__(self, *args, action=''):
        super().__init__(*args)
        self.action = action

    def __str__(self):
        message = super().__str__()
        if not message:
            message = self.default_message
        desc = f"Code: {self.code}, Message: {message}"
        if self.action:
            desc += f"\nTIP: {self.action}"
        return desc


# EnvironmentError
EnvError: Type[ModelslimError] = ModelslimError.create_exception("EnvError", 100,
                                                                 "Environment failed to meet the requirements.")
VersionError: Type[ModelslimError] = EnvError.create_exception("VersionError", 101,
                                                               "Version of dependencies mismatched.")
EnvVarError: Type[ModelslimError] = EnvError.create_exception("EnvVarError", 102,
                                                              "Environment variable not set right.")
ConfigError: Type[ModelslimError] = EnvError.create_exception("ConfigError", 103,
                                                              "Config file is invalid.")

# MisbehaviorError
MisbehaviorError: Type[ModelslimError] = ModelslimError.create_exception("MisbehaviorError", 200, "User misbehavior.")
InvalidModelError: Type[ModelslimError] = MisbehaviorError.create_exception("InvalidModelError", 201,
                                                                            "Invalid model to load or inference.")
InvalidDatasetError: Type[ModelslimError] = MisbehaviorError.create_exception("InvalidDatasetError", 202,
                                                                              "Invalid dataset to load.")
SchemaValidateError: Type[ModelslimError] = MisbehaviorError.create_exception("SchemaValidateError", 203,
                                                                              "Argument schema validation failed.")
SecurityError: Type[ModelslimError] = MisbehaviorError.create_exception("SecurityError", 204,
                                                                        "Potential security risk.")

# TrivialError
TrivialError: Type[ModelslimError] = ModelslimError.create_exception("TrivialError", 300,
                                                                     "Trivial error, no need to resolve.")
UnsupportedError: Type[ModelslimError] = TrivialError.create_exception("UnsupportedError", 301,
                                                                       "Unsupported operation.")
SpecError: Type[ModelslimError] = TrivialError.create_exception("SpecError", 302,
                                                                "Specific scenario error.")

# ToDoError
ToDoError: Type[ModelslimError] = ModelslimError.create_exception("ToDoError", 400, "Bug to be fixed soon.")

# UnexpectedError
UnexpectedError: Type[ModelslimError] = ModelslimError.create_exception("UnexpectedError", 500, "Unexpected error.")

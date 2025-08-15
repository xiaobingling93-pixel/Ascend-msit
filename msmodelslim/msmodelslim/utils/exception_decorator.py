# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from functools import wraps

from typing_extensions import Type

from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.exception import UnexpectedError, ModelslimError
from msmodelslim.utils.logging import get_logger

ACTION_REPORT = f'Please report this issue to the msModelSlim developers.' \
                f'{msmodelslim_config.urls.repository}'


def exception_handler(*set_args,
                      err_cls: Type[Exception] = Exception,
                      ms_err_cls: Type[ModelslimError] = ModelslimError,
                      keyword: str = '', action: str = ''):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ModelslimError:
                raise
            except err_cls as e:
                if not keyword or keyword in str(e):
                    raise ms_err_cls(*set_args if set_args else str(e), action=action) from e
                raise
            except Exception:
                raise

        return wrapper

    return decorator


def exception_catcher(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ModelslimError as e:
            get_logger().error(f'{repr(e)}')
            raise
        except Exception as e:
            get_logger().error(f'{repr(e)}')
            raise UnexpectedError(action=ACTION_REPORT) from e

    return wrapper

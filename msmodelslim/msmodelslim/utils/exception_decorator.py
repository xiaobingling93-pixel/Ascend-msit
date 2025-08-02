# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from functools import wraps
from logging import Logger

from typing_extensions import Type, Union

from msmodelslim.utils.exception import UnexpectedError, ModelslimError

from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.logger import get_logger

ACTION_REPORT = f'Please report this issue to the msModelSlim developers.' \
                f'{msmodelslim_config.urls.repository}'


def exception_handler(err_cls: Type[Exception], ms_err_cls: Type[ModelslimError], keyword: str = '', message: str = '',
                      action: str = ''):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ModelslimError:
                raise
            except err_cls as e:
                if not keyword or keyword in str(e):
                    raise ms_err_cls(message if message else str(e), action=action) from e
                raise UnexpectedError(action=ACTION_REPORT) from e
            except Exception as e:
                raise UnexpectedError(action=ACTION_REPORT) from e

        return wrapper

    return decorator


def exception_catcher(logger: Union[Logger, str]):
    if isinstance(logger, str):
        logger = get_logger(logger)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ModelslimError as e:
                logger.error(f'{repr(e)}')
                raise
            except Exception as e:
                logger.error(f'{repr(e)}')
                raise UnexpectedError(action=ACTION_REPORT) from e

        return wrapper

    return decorator

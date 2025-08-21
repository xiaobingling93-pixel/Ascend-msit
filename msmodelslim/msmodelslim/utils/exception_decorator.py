# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import traceback
from functools import wraps

from typing_extensions import Type

from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.exception import UnexpectedError, ModelslimError
from msmodelslim.utils.logging import get_logger

ACTION_REPORT = f'Please report this issue to the msModelSlim developers. ' \
                f'Repository: {msmodelslim_config.urls.repository} ' \
                f'Q&A: {msmodelslim_config.urls.question_and_answer}'


def exception_handler(*set_args,
                      err_cls: Type[Exception] = Exception,
                      ms_err_cls: Type[ModelslimError] = ModelslimError,
                      keyword: str = '', action: str = ''):
    """
    异常知识库构建装饰器，用于将第三方异常的经验和解决方案沉淀到代码中。
    
    该装饰器主要用于构建异常知识库，将已知的第三方异常（如文件不存在、网络超时、权限错误等）
    转换为msModelSlim自定义异常，并提供针对性的解决建议。通过这种方式，将开发运维过程中
    积累的异常处理经验固化到代码中，为用户提供更好的错误提示和解决指导。
    
    Args:
        *set_args: 传递给自定义异常的参数，通常包含具体的错误描述
        err_cls: 要捕获的第三方异常类型，如FileNotFoundError、TimeoutError等
        ms_err_cls: 转换后的msModelSlim异常类型，如ConfigError、InvalidModelError等
        keyword: 异常消息中必须包含的关键字，用于精确匹配特定的异常场景
        action: 针对性的解决建议，如"请检查文件路径"、"请检查网络连接"等
    
    Returns:
        装饰器函数
    
    Example:
        # 文件操作异常知识库
        @exception_handler("配置文件不存在", ms_err_cls=ConfigError,
                          err_cls=FileNotFoundError,
                          keyword="No such file",
                          action="请检查配置文件路径是否正确")
        def load_config():
            # 加载配置文件的代码
            pass
        
        # 权限异常知识库
        @exception_handler("权限不足", ms_err_cls=SecurityError,
                          err_cls=PermissionError,
                          action="请使用管理员权限运行或检查文件权限")
        def write_output():
            # 写入输出文件的代码
            pass
        
        # 内存不足异常知识库
        @exception_handler("内存不足", ms_err_cls=ResourceError,
                          keyword="out of memory",
                          action="请减少批处理大小或增加系统内存")
        def process_large_data():
            # 处理大数据集的代码
            pass
    
    Note:
        - 主要用于构建异常知识库，将运维经验沉淀到代码中
        - 通过精确的异常匹配和针对性的解决建议，提升用户体验
        - 随着项目发展，可以不断丰富异常知识库的内容
        - 建议为每种常见的第三方异常场景都建立相应的知识库条目
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ModelslimError:
                # 如果已经是msModelSlim异常，直接抛出
                raise
            except err_cls as e:
                # 检查是否包含指定关键字，如果指定了关键字的话
                if not keyword or keyword in str(e):
                    # set_args 为空时，确保作为单个参数传入而非按字符拆分
                    args_to_raise = set_args if set_args else (str(e),)
                    raise ms_err_cls(*args_to_raise, action=action) from e
                raise
            except Exception:
                # 其他类型的异常直接抛出
                raise

        return wrapper

    return decorator


def exception_catcher(func):
    """
    应用级异常捕获装饰器，用于统一处理函数中的异常并记录日志。
    
    该装饰器主要用于应用入口或顶层函数，作为兜底机制捕获所有未被处理的异常，
    记录详细的错误日志，并将非msModelSlim异常转换为UnexpectedError，同时提供标准的错误报告信息。
    
    Args:
        func: 要装饰的函数
    
    Returns:
        装饰后的函数
    
    Example:
        # 应用入口使用：兜底捕获所有异常
        @exception_catcher
        def main():
            # 应用主逻辑
            pass
        
        # 顶层API函数使用
        @exception_catcher
        def api_endpoint():
            # API处理逻辑
            pass
        
        # 与组件级装饰器组合使用（exception_catcher应该在最外层）
        @exception_catcher
        @exception_handler(err_cls=ValueError, ms_err_cls=ConfigError)
        def complex_workflow():
            # 复杂的工作流程
            pass
    
    Note:
        - 适用于应用级别的兜底异常处理
        - 会记录所有异常的详细信息到日志中，包括完整的错误栈
        - 对于非msModelSlim异常，会转换为UnexpectedError并包含标准的错误报告链接
        - 建议在应用入口、API端点或顶层函数上使用此装饰器
        - 当与exception_handler组合使用时，exception_catcher应该放在最外层
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ModelslimError as e:
            # 记录msModelSlim异常
            get_logger().error(f'{repr(e)}')
            raise
        except Exception as e:
            # 先转换为UnexpectedError，再记录日志和错误栈
            unexpected_error = UnexpectedError(action=ACTION_REPORT)
            get_logger().error(f'{repr(unexpected_error)}')
            get_logger().error(f'Original exception: {repr(e)}')
            get_logger().error(f'Traceback:\n{traceback.format_exc()}')
            raise unexpected_error from e

    return wrapper

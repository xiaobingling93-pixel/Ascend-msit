import sys
import importlib
import inspect
import logging
import functools
import threading
from packaging.version import Version
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 全局钩子注册表（线程安全）
_HOOK_REGISTRY: Dict[str, List['HookHelper']] = {}
_registry_lock = threading.Lock()

class HookHelper:
    def __init__(
        self,
        original_func: Callable,
        new_func: Callable,
        location: Any,
        attr_name: str
    ):
        self.original_func = original_func
        self.new_func = new_func
        self.location = location
        self.attr_name = attr_name
        self._orig_descriptor = self._get_original_descriptor()

    @classmethod
    def create(
        cls,
        original_func_define: Callable,
        new_func: Callable
    ) -> Optional['HookHelper']:
        """工厂方法创建HookHelper实例"""
        if original_func_define is None:
            return None
            
        # 获取原始函数和位置信息
        original_func = cls._resolve_callable(original_func_define)
        if not original_func:
            logging.error("Invalid callable: %s", original_func_define)
            return None
            
        location, attr_name = cls.get_location(original_func)
        if not all((original_func, location, attr_name)):
            return None
            
        return cls(original_func, new_func, location, attr_name)

    @staticmethod
    def _resolve_callable(obj: Callable) -> Optional[Callable]:
        """解析可调用对象的核心函数"""
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            return obj
        if callable(obj):
            return obj.__call__
        return None

    @staticmethod
    def get_location(func: Callable) -> Tuple[Any, str]:
        """获取函数位置信息，支持类方法"""
        if not hasattr(func, "__module__"):
            raise AttributeError(f"Function {func} missing __module__ attribute")
        
        try:
            module = importlib.import_module(func.__module__)
        except ImportError as e:
            logging.error("Module import failed: %s", e)
            raise
            
        qual_parts = func.__qualname__.split(".")
        *class_path, attr_name = qual_parts
        
        # 遍历类层次结构
        current = module
        for part in class_path:
            current = getattr(current, part, None)
            if current is None:
                # 尝试动态导入类
                try:
                    class_module = importlib.import_module(func.__module__)
                    current = getattr(class_module, part)
                except (ImportError, AttributeError) as e:
                    raise AttributeError(f"Class {part} not found: {e}")
                
        return current, attr_name

    def _get_original_descriptor(self) -> Any:
        """获取原始描述符类型（类方法/静态方法）"""
        if not inspect.isclass(self.location):
            return None
            
        try:
            # 使用安全方法获取属性
            attr = getattr(self.location, self.attr_name)
            if isinstance(attr, staticmethod):
                return staticmethod
            elif isinstance(attr, classmethod):
                return classmethod
            return None
        except AttributeError:
            return None

    def replace(self) -> None:
        """替换原始函数，正确处理类方法和静态方法"""
        if self._orig_descriptor is staticmethod:
            setattr(self.location, self.attr_name, staticmethod(self.new_func))
        elif self._orig_descriptor is classmethod:
            setattr(self.location, self.attr_name, classmethod(self.new_func))
        else:
            setattr(self.location, self.attr_name, self.new_func)

    def recover(self) -> None:
        """恢复原始函数，正确处理类方法和静态方法"""
        if self._orig_descriptor is staticmethod:
            setattr(self.location, self.attr_name, staticmethod(self.original_func))
        elif self._orig_descriptor is classmethod:
            setattr(self.location, self.attr_name, classmethod(self.original_func))
        else:
            setattr(self.location, self.attr_name, self.original_func)

def _import_target_module(module_path: str) -> Optional[ModuleType]:
    """安全导入目标模块"""
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        logging.error("Module import failed for %s: %s", module_path, e)
        return None

def _resolve_target_function(
    module: ModuleType, 
    function_path: str
) -> Optional[Tuple[Callable, Any, str]]:
    """解析目标函数，支持类方法"""
    parts = function_path.split('.')
    target = module
    
    # 遍历路径直到最后一个属性
    for part in parts[:-1]:
        target = getattr(target, part, None)
        if target is None:
            logging.error("Attribute %s not found in %s", part, function_path)
            return None
            
    attr_name = parts[-1]
    original_func = getattr(target, attr_name, None)
    
    if original_func is None:
        logging.error("Function %s not found in module", function_path)
        return None
        
    return original_func, target, attr_name

def _create_wrapper(
    original_func: Callable, 
    user_func: Callable, 
    caller_filter: Optional[str]
) -> Callable:
    """创建安全的包装函数"""
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        # 调用栈过滤
        if caller_filter:
            frame = sys._getframe(1)
            if frame.f_code.co_name != caller_filter:
                return original_func(*args, **kwargs)
        
        try:
            return user_func(original_func, *args, **kwargs)
        except Exception as e:
            logging.error("Hook function error: %s", e, exc_info=True)
            return original_func(*args, **kwargs)
            
    return wrapper

def _check_version_compatibility(
    min_version: Optional[str], 
    max_version: Optional[str]
) -> bool:
    """检查版本兼容性"""
    try:
        from vllm import __version__ as current_version
        current_ver = Version(current_version)

        if min_version and current_ver < Version(min_version):
            logging.info("Skipping hooks: Current version %s < %s", current_version, min_version)
            return False
        if max_version and current_ver > Version(max_version):
            logging.info("Skipping hooks: Current version %s > %s", current_version, max_version)
            return False
        return True
    except ImportError:
        logging.warning("VLLM version check failed - proceeding with hooks")
        return True

def _install_hook(
    hook_point: str, 
    user_func: Callable, 
    caller_filter: Optional[str]
) -> Optional[HookHelper]:
    """安装单个钩子，增强类方法处理"""
    module_path, _, function_path = hook_point.rpartition('.')
    if not module_path:
        logging.error("Invalid hook point format: %s", hook_point)
        return None

    module = _import_target_module(module_path)
    if not module:
        return None

    target_info = _resolve_target_function(module, function_path)
    if not target_info:
        return None
        
    original_func, target_obj, attr_name = target_info
    
    # 创建安全的包装器
    wrapper = _create_wrapper(original_func, user_func, caller_filter)
    
    try:
        helper = HookHelper.create(original_func, wrapper)
        if helper:
            helper.replace()
            logging.info("Hook installed: %s", hook_point)
            return helper
    except Exception as e:
        logging.error("Hook installation failed for %s: %s", hook_point, e)
        
    return None

def vllm_hook(
    hook_points: Union[str, List[str]],
    caller_filter: Optional[str] = None,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
) -> Callable:
    """主装饰器函数"""
    # 标准化钩子点
    targets = [hook_points] if isinstance(hook_points, str) else hook_points
    
    def decorator(user_func: Callable) -> Callable:
        # 版本检查
        if (min_version or max_version) and not _check_version_compatibility(min_version, max_version):
            return user_func
            
        helpers = []
        
        # 安装所有钩子
        for point in targets:
            helper = _install_hook(point, user_func, caller_filter)
            if helper:
                helpers.append(helper)
                
        # 注册钩子以便后续恢复
        if helpers:
            with _registry_lock:
                key = f"{user_func.__module__}.{user_func.__name__}"
                _HOOK_REGISTRY.setdefault(key, []).extend(helpers)
                
        return user_func
        
    return decorator

def recover_hooks_for(func: Callable) -> None:
    """恢复特定函数的所有钩子"""
    key = f"{func.__module__}.{func.__name__}"
    with _registry_lock:
        helpers = _HOOK_REGISTRY.pop(key, [])
        
    for helper in helpers:
        try:
            helper.recover()
            logging.info("Hook recovered: %s.%s", 
                         helper.location.__name__ if hasattr(helper.location, '__name__') else type(helper.location).__name__, 
                         helper.attr_name)
        except Exception as e:
            logging.error("Hook recovery failed: %s", e)

def recover_all_hooks() -> None:
    """恢复所有已注册的钩子"""
    with _registry_lock:
        all_helpers = []
        for helpers in _HOOK_REGISTRY.values():
            all_helpers.extend(helpers)
        _HOOK_REGISTRY.clear()
        
    for helper in all_helpers:
        try:
            helper.recover()
            logging.info("Hook recovered: %s.%s", 
                         helper.location.__name__ if hasattr(helper.location, '__name__') else type(helper.location).__name__, 
                         helper.attr_name)
        except Exception as e:
            logging.error("Hook recovery failed: %s", e)
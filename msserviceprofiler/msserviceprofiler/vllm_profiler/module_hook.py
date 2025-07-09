import sys
import importlib
import inspect
import logging
import functools
import threading
from packaging.version import Version
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable

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
        logging.debug("Created HookHelper for %s.%s", 
                      location.__name__ if hasattr(location, '__name__') else type(location).__name__, 
                      attr_name)

    @staticmethod
    def _get_descriptor_type(obj: Any) -> Optional[type]:
        """获取描述符类型（类方法/静态方法）"""
        if isinstance(obj, staticmethod):
            return staticmethod
        if isinstance(obj, classmethod):
            return classmethod
        return None

    def _get_original_descriptor(self) -> Optional[type]:
        """获取原始描述符类型"""
        if not inspect.isclass(self.location):
            return None
            
        try:
            # 使用安全方法获取属性
            attr = getattr(self.location, self.attr_name)
            return self._get_descriptor_type(attr)
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
        logging.info("Replaced function: %s.%s", 
                     self.location.__name__ if hasattr(self.location, '__name__') else type(self.location).__name__, 
                     self.attr_name)

    def recover(self) -> None:
        """恢复原始函数，正确处理类方法和静态方法"""
        if self._orig_descriptor is staticmethod:
            setattr(self.location, self.attr_name, staticmethod(self.original_func))
        elif self._orig_descriptor is classmethod:
            setattr(self.location, self.attr_name, classmethod(self.original_func))
        else:
            setattr(self.location, self.attr_name, self.original_func)
        logging.info("Recovered function: %s.%s", 
                     self.location.__name__ if hasattr(self.location, '__name__') else type(self.location).__name__, 
                     self.attr_name)

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
    """解析目标函数，支持嵌套属性和类方法"""
    parts = function_path.split('.')
    current = module
    
    # 遍历路径直到最后一个属性
    for i, part in enumerate(parts):
        try:
            # 尝试获取当前属性
            attr = getattr(current, part)
            
            # 如果是最后一个部分，直接返回
            if i == len(parts) - 1:
                return attr, current, part
                
            # 否则继续深入
            current = attr
        except AttributeError:
            # 尝试导入子模块（对于嵌套模块情况）
            submodule_path = f"{current.__name__}.{part}" if hasattr(current, '__name__') else part
            try:
                submodule = importlib.import_module(submodule_path)
                current = submodule
            except ImportError:
                logging.error("Attribute %s not found in %s", part, function_path)
                return None
                
    return None

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
    module_path: str,
    function_path: str,
    user_func: Callable, 
    caller_filter: Optional[str]
) -> Optional[HookHelper]:
    """安装单个钩子"""
    module = _import_target_module(module_path)
    if not module:
        logging.error("Module not found: %s", module_path)
        return None

    # 解析目标函数
    target_info = _resolve_target_function(module, function_path)
    if target_info is None:
        logging.error("Function not found: %s in module %s", function_path, module_path)
        return None
        
    original_func, location, attr_name = target_info
    
    # 创建安全的包装器
    wrapper = _create_wrapper(original_func, user_func, caller_filter)
    
    try:
        helper = HookHelper(original_func, wrapper, location, attr_name)
        helper.replace()
        logging.info("Hook installed: %s.%s", module_path, function_path)
        return helper
    except Exception as e:
        logging.error("Hook installation failed for %s.%s: %s", module_path, function_path, e)
        return None

HookPoint = Union[Tuple[str, str], List[Tuple[str, str]]]

def vllm_hook(
    hook_points: HookPoint,
    caller_filter: Optional[str] = None,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
) -> Callable:
    """主装饰器函数，支持元组形式钩子点"""
    # 标准化钩子点
    if isinstance(hook_points, tuple):
        targets = [hook_points]
    elif isinstance(hook_points, list) and all(isinstance(hp, tuple) for hp in hook_points):
        targets = hook_points
    else:
        raise TypeError("hook_points must be a tuple (module_path, function_path) or list of tuples")
    
    def decorator(user_func: Callable) -> Callable:
        # 版本检查
        if (min_version or max_version) and not _check_version_compatibility(min_version, max_version):
            logging.info("Skipping hooks due to version incompatibility")
            return user_func
            
        helpers = []
        
        # 安装所有钩子
        for module_path, function_path in targets:
            helper = _install_hook(module_path, function_path, user_func, caller_filter)
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
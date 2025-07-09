import os
import pytest
import sys
import importlib
import pkgutil
import threading
from types import ModuleType
from unittest.mock import patch, MagicMock
from packaging.version import Version

sys.path.append(os.path.join(os.path.dirname(pkgutil.get_loader("msserviceprofiler").path), "vllm_profiler"))
from vllm_profiler.module_hook import (
    HookHelper,
    _import_target_module,
    _resolve_target_function,
    _create_wrapper,
    _check_version_compatibility,
    _install_hook,
    vllm_hook,
    recover_hooks_for,
    recover_all_hooks,
    _HOOK_REGISTRY,
    _registry_lock,
)

# Test Helpers
class TestClass:
    @staticmethod
    def static_method():
        return "original_static"

    @classmethod
    def class_method(cls):
        return "original_class"

    def instance_method(self):
        return "original_instance"

def dummy_user_func(original_func, *args, **kwargs):
    return f"hooked_{original_func(*args, **kwargs)}"

# HookHelper Tests
def test_HookHelper_init_given_regular_function_when_initialized_then_sets_properties_correctly():
    """Test HookHelper initialization with regular function"""
    original = lambda: None
    new = lambda: None
    location = TestClass
    helper = HookHelper(original, new, location, "test_method")
    
    assert helper.original_func == original
    assert helper.new_func == new
    assert helper.location == location
    assert helper.attr_name == "test_method"
    assert helper._orig_descriptor is None

def test_HookHelper_init_given_staticmethod_when_initialized_then_detects_descriptor():
    """Test HookHelper initialization with staticmethod"""
    helper = HookHelper(TestClass.static_method.__func__, lambda: None, TestClass, "static_method")
    assert helper._orig_descriptor is staticmethod

def test_HookHelper_init_given_classmethod_when_initialized_then_detects_descriptor():
    """Test HookHelper initialization with classmethod"""
    helper = HookHelper(TestClass.class_method.__func__, lambda: None, TestClass, "class_method")
    assert helper._orig_descriptor is classmethod

def test_HookHelper_replace_given_staticmethod_when_replaced_then_preserves_descriptor():
    """Test replacing a staticmethod preserves descriptor"""
    original = TestClass.static_method
    helper = HookHelper(original.__func__, lambda: "new", TestClass, "static_method")
    helper.replace()
    
    assert isinstance(getattr(TestClass, "static_method"), staticmethod)
    assert TestClass.static_method() == "new"

def test_HookHelper_replace_given_classmethod_when_replaced_then_preserves_descriptor():
    """Test replacing a classmethod preserves descriptor"""
    original = TestClass.class_method
    helper = HookHelper(original.__func__, lambda cls: "new", TestClass, "class_method")
    helper.replace()
    
    assert isinstance(getattr(TestClass, "class_method"), classmethod)
    assert TestClass.class_method() == "new"

def test_HookHelper_replace_given_regular_method_when_replaced_then_sets_new_function():
    """Test replacing regular method"""
    original = TestClass.instance_method
    helper = HookHelper(original, lambda self: "new", TestClass, "instance_method")
    helper.replace()
    
    assert TestClass().instance_method() == "new"

def test_HookHelper_recover_given_staticmethod_when_recovered_then_restores_original():
    """Test recovering staticmethod"""
    original = TestClass.static_method
    helper = HookHelper(original.__func__, lambda: "new", TestClass, "static_method")
    helper.replace()
    helper.recover()
    
    assert TestClass.static_method() == "original_static"

# _import_target_module Tests
def test__import_target_module_given_existing_module_when_imported_then_returns_module():
    """Test importing existing module"""
    module = _import_target_module("sys")
    assert isinstance(module, ModuleType)
    assert module.__name__ == "sys"

def test__import_target_module_given_nonexistent_module_when_imported_then_returns_none():
    """Test importing non-existent module"""
    module = _import_target_module("nonexistent_module_123")
    assert module is None

# _resolve_target_function Tests
def test__resolve_target_function_given_module_function_when_resolved_then_returns_correct_info():
    """Test resolving function in module"""
    module = importlib.import_module("sys")
    func_info = _resolve_target_function(module, "exit")
    
    assert func_info is not None
    func, loc, name = func_info
    assert func == sys.exit
    assert loc == sys
    assert name == "exit"

def test__resolve_target_function_given_nested_attribute_when_resolved_then_returns_correct_info():
    """Test resolving nested attribute"""
    class A:
        class B:
            @staticmethod
            def func():
                return "test"
    
    module = MagicMock()
    module.A = A
    func_info = _resolve_target_function(module, "A.B.func")
    
    assert func_info is not None
    func, loc, name = func_info
    assert func() == "test"
    assert loc == A.B
    assert name == "func"

def test__resolve_target_function_given_nonexistent_function_when_resolved_then_returns_none():
    """Test resolving non-existent function"""
    module = importlib.import_module("sys")
    func_info = _resolve_target_function(module, "nonexistent_func")
    assert func_info is None

# _create_wrapper Tests
def test__create_wrapper_given_no_filter_when_called_then_calls_user_func():
    """Test wrapper without caller filter"""
    def original():
        return "original"
    
    wrapper = _create_wrapper(original, dummy_user_func, None)
    assert wrapper() == "hooked_original"

def test__create_wrapper_given_filter_mismatch_when_called_then_calls_original():
    """Test wrapper with mismatched caller filter"""
    def original():
        return "original"
    
    wrapper = _create_wrapper(original, dummy_user_func, "wrong_caller")
    
    with patch.object(sys, '_getframe', return_value=MagicMock(f_code=MagicMock(co_name="actual_caller"))):
        assert wrapper() == "original"

def test__create_wrapper_given_filter_match_when_called_then_calls_user_func():
    """Test wrapper with matching caller filter"""
    def original():
        return "original"
    
    wrapper = _create_wrapper(original, dummy_user_func, "correct_caller")
    
    with patch.object(sys, '_getframe', return_value=MagicMock(f_code=MagicMock(co_name="correct_caller"))):
        assert wrapper() == "hooked_original"

def test__create_wrapper_given_user_func_error_when_called_then_calls_original():
    """Test wrapper handles user function errors"""
    def original():
        return "original"
    
    def failing_user_func(*args, **kwargs):
        raise ValueError("Error")
    
    wrapper = _create_wrapper(original, failing_user_func, None)
    assert wrapper() == "original"

# _check_version_compatibility Tests
@patch('vllm_profiler.module_hook.logger')
def test__check_version_compatibility_given_no_version_constraints_when_checked_then_returns_true(mock_logger):
    """Test version check with no constraints"""
    assert _check_version_compatibility(None, None) is True
    mock_logger.warning.assert_not_called()

@patch('vllm_profiler.module_hook.logger')
@patch('vllm_profiler.module_hook.Version')
def test__check_version_compatibility_given_min_version_when_checked_then_returns_correct_result(mock_version, mock_logger):
    """Test version check with min version"""
    mock_version.return_value = Version("1.0")
    assert _check_version_compatibility("0.9", None) is True
    assert _check_version_compatibility("1.1", None) is False

@patch('vllm_profiler.module_hook.logger')
@patch('vllm_profiler.module_hook.Version')
def test__check_version_compatibility_given_max_version_when_checked_then_returns_correct_result(mock_version, mock_logger):
    """Test version check with max version"""
    mock_version.return_value = Version("1.0")
    assert _check_version_compatibility(None, "1.1") is True
    assert _check_version_compatibility(None, "0.9") is False

# _install_hook Tests
def test__install_hook_given_valid_function_when_installed_then_returns_helper():
    """Test installing hook on valid function"""
    helper = _install_hook("sys", "exit", dummy_user_func, None)
    assert isinstance(helper, HookHelper)
    # Cleanup
    helper.recover()

def test__install_hook_given_nonexistent_module_when_installed_then_returns_none():
    """Test installing hook on non-existent module"""
    helper = _install_hook("nonexistent_module", "func", dummy_user_func, None)
    assert helper is None

def test__install_hook_given_nonexistent_function_when_installed_then_returns_none():
    """Test installing hook on non-existent function"""
    helper = _install_hook("sys", "nonexistent_func", dummy_user_func, None)
    assert helper is None

# vllm_hook Tests
def test_vllm_hook_given_single_hook_point_when_decorated_then_installs_hook():
    """Test decorator with single hook point"""
    @vllm_hook(("sys", "exit"))
    def hook_func(original, *args, **kwargs):
        return "hooked"
    
    # Verify hook was registered
    key = f"{hook_func.__module__}.{hook_func.__name__}"
    assert key in _HOOK_REGISTRY
    assert len(_HOOK_REGISTRY[key]) == 1
    # Cleanup
    recover_hooks_for(hook_func)

def test_vllm_hook_given_multiple_hook_points_when_decorated_then_installs_all_hooks():
    """Test decorator with multiple hook points"""
    @vllm_hook([("sys", "exit"), ("sys", "getsizeof")])
    def hook_func(original, *args, **kwargs):
        return "hooked"
    
    # Verify hooks were registered
    key = f"{hook_func.__module__}.{hook_func.__name__}"
    assert key in _HOOK_REGISTRY
    assert len(_HOOK_REGISTRY[key]) == 2
    # Cleanup
    recover_hooks_for(hook_func)

def test_vllm_hook_given_invalid_hook_points_when_decorated_then_raises_typeerror():
    """Test decorator with invalid hook points"""
    with pytest.raises(TypeError):
        @vllm_hook("invalid_hook_points")
        def hook_func():
            pass

def test_vllm_hook_given_version_mismatch_when_decorated_then_skips_hooks():
    """Test decorator with version mismatch"""
    @vllm_hook(("sys", "exit"), min_version="999.0")
    def hook_func(original, *args, **kwargs):
        return "hooked"
    
    # Verify no hooks were registered
    key = f"{hook_func.__module__}.{hook_func.__name__}"
    assert key not in _HOOK_REGISTRY

# recover_hooks_for Tests
def test_recover_hooks_for_given_registered_function_when_called_then_removes_hooks():
    """Test recovering hooks for specific function"""
    @vllm_hook(("sys", "exit"))
    def hook_func(original, *args, **kwargs):
        return "hooked"
    
    key = f"{hook_func.__module__}.{hook_func.__name__}"
    assert key in _HOOK_REGISTRY
    
    recover_hooks_for(hook_func)
    assert key not in _HOOK_REGISTRY

def test_recover_hooks_for_given_unregistered_function_when_called_then_does_nothing():
    """Test recovering hooks for unregistered function"""
    def unregistered_func():
        pass
    
    recover_hooks_for(unregistered_func)
    # No error should occur

# recover_all_hooks Tests
def test_recover_all_hooks_given_registered_hooks_when_called_then_clears_registry():
    """Test recovering all hooks"""
    @vllm_hook(("sys", "exit"))
    def hook_func1(original, *args, **kwargs):
        return "hooked"
    
    @vllm_hook(("sys", "getsizeof"))
    def hook_func2(original, *args, **kwargs):
        return "hooked"
    
    assert len(_HOOK_REGISTRY) == 2
    recover_all_hooks()
    assert len(_HOOK_REGISTRY) == 0

# Thread Safety Tests
def test_registry_thread_safety_given_concurrent_access_when_modified_then_maintains_integrity():
    """Test thread safety of hook registry"""
    def register_hooks():
        with _registry_lock:
            if "test_key" not in _HOOK_REGISTRY:
                _HOOK_REGISTRY["test_key"] = []
            _HOOK_REGISTRY["test_key"].append("dummy_helper")

    threads = [threading.Thread(target=register_hooks) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(_HOOK_REGISTRY["test_key"]) == 10
    # Cleanup
    _HOOK_REGISTRY.clear()
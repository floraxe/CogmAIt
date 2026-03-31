import importlib
import importlib.util
import inspect
import os
import sys
import time
import threading
import pkgutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Set

from fastapi import HTTPException
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileDeletedEvent, FileModifiedEvent

from app.core.config import settings
from app.providers.base import ModelProvider

# 设置watchfiles日志级别为ERROR，减少不必要的日志
logging.getLogger('watchfiles').setLevel(logging.ERROR)

# MCP提供商注册表
_mcp_providers = {}

def register_mcp_provider(provider_id: str, provider_class):
    """注册MCP服务提供商"""
    global _mcp_providers
    _mcp_providers[provider_id] = provider_class

def get_mcp_provider(provider_id: str):
    """获取MCP服务提供商"""
    global _mcp_providers
    if provider_id not in _mcp_providers:
        return None
    return _mcp_providers[provider_id]()

class ProviderFileHandler(FileSystemEventHandler):
    """
    监听提供商文件变化的处理器
    """
    
    def __init__(self, manager):
        self.manager = manager
        # 记录最近处理的事件时间，避免短时间内重复处理
        self._last_event_time = {}
    
    def _should_process_event(self, event):
        """检查是否应该处理此事件"""
        # 基本检查：必须是.py文件且不在排除列表中
        if not event.is_directory and event.src_path.endswith('.py'):
            filename = os.path.basename(event.src_path)
            module_name = os.path.splitext(filename)[0]
            
            # 排除基础模块和管理器模块
            if module_name not in ["base", "manager", "__init__"] and not module_name.startswith("__"):
                # 防止短时间内重复处理同一文件
                current_time = time.time()
                last_time = self._last_event_time.get(event.src_path, 0)
                if current_time - last_time > 1.0:  # 至少间隔1秒
                    self._last_event_time[event.src_path] = current_time
                    return True
        return False
    
    def on_created(self, event):
        """当新文件被创建时"""
        if self._should_process_event(event):
            filename = os.path.basename(event.src_path)
            module_name = os.path.splitext(filename)[0]
            print(f"检测到新提供商模块: {module_name}")
            self.manager._load_provider_module(module_name)
    
    def on_deleted(self, event):
        """当文件被删除时"""
        if self._should_process_event(event):
            filename = os.path.basename(event.src_path)
            module_name = os.path.splitext(filename)[0]
            print(f"检测到提供商模块被删除: {module_name}")
            self.manager._unload_provider_module(module_name)
    
    def on_modified(self, event):
        """当文件被修改时"""
        if self._should_process_event(event):
            filename = os.path.basename(event.src_path)
            module_name = os.path.splitext(filename)[0]
            print(f"检测到提供商模块被修改: {module_name}")
            self.manager._reload_provider_module(module_name)


class ProviderManager:
    """
    模型提供商管理器
    
    负责发现、加载和管理模型提供商插件，支持动态监测和加载
    """
    
    def __init__(self):
        self._providers: Dict[str, ModelProvider] = {}
        self._loaded_modules: Set[str] = set()
        self._provider_classes: Dict[str, Dict[str, Type[ModelProvider]]] = {}
        self._package_dir = ""
        self._observer = None
        self._lock = threading.RLock()  # 使用可重入锁确保线程安全
        self._load_providers()
        self._start_watcher()
    
    def _load_providers(self) -> None:
        """
        动态加载所有可用的模型提供商插件
        """
        with self._lock:
            provider_package = settings.PROVIDERS_PACKAGE
            try:
                package = importlib.import_module(provider_package)
                self._package_dir = os.path.dirname(package.__file__)
                
                # 遍历包中的所有模块
                for _, module_name, is_pkg in pkgutil.iter_modules([self._package_dir]):
                    if module_name not in ["base", "manager", "__init__"] and not is_pkg and not module_name.startswith("__"):
                        self._load_provider_module(module_name)
                
                print(f"已加载 {len(self._providers)} 个模型提供商")
            except Exception as e:
                print(f"加载模型提供商时出错: {str(e)}")
    
    def _load_provider_module(self, module_name: str) -> None:
        """
        加载单个提供商模块
        
        参数:
            module_name (str): 模块名称
        """
        with self._lock:
            provider_package = settings.PROVIDERS_PACKAGE
            
            try:
                # 导入模块
                full_module_name = f"{provider_package}.{module_name}"
                
                # 如果模块已在sys.modules中，先移除它以确保重新加载
                if full_module_name in sys.modules:
                    del sys.modules[full_module_name]
                    
                module = importlib.import_module(full_module_name)
                
                # 存储模块中的提供商类
                self._provider_classes[module_name] = {}
                
                # 查找继承自ModelProvider的类
                providers_found = 0
                for class_name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, ModelProvider) 
                        and obj != ModelProvider 
                        and hasattr(obj, "provider_id")
                    ):
                        # 保存提供商类引用
                        self._provider_classes[module_name][class_name] = obj
                        
                        # 实例化提供商并添加到字典
                        provider = obj()
                        self._providers[provider.provider_id] = provider
                        providers_found += 1
                        print(f"已加载提供商: {provider.provider_name} ({provider.provider_id})")
                
                if providers_found > 0:
                    # 记录已加载的模块
                    self._loaded_modules.add(module_name)
                else:
                    print(f"警告: 模块 {module_name} 中未找到提供商类")
                    
            except Exception as e:
                print(f"加载提供商模块 {module_name} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _unload_provider_module(self, module_name: str) -> None:
        """
        卸载提供商模块
        
        参数:
            module_name (str): 模块名称
        """
        with self._lock:
            try:
                # 检查模块是否已加载
                if module_name not in self._loaded_modules:
                    return
                
                # 获取该模块中的所有提供商
                if module_name in self._provider_classes:
                    # 移除该模块中的所有提供商实例
                    providers_to_remove = []
                    for provider_id, provider in self._providers.items():
                        provider_class = provider.__class__.__name__
                        if module_name in self._provider_classes and provider_class in self._provider_classes[module_name]:
                            providers_to_remove.append(provider_id)
                    
                    # 从字典中移除提供商
                    for provider_id in providers_to_remove:
                        if provider_id in self._providers:
                            print(f"卸载提供商: {self._providers[provider_id].provider_name} ({provider_id})")
                            del self._providers[provider_id]
                    
                    # 清理模块的类记录
                    del self._provider_classes[module_name]
                
                # 从已加载模块集合中移除
                self._loaded_modules.remove(module_name)
                
                # 从sys.modules中移除，确保下次能重新加载
                full_module_name = f"{settings.PROVIDERS_PACKAGE}.{module_name}"
                if full_module_name in sys.modules:
                    del sys.modules[full_module_name]
                
                print(f"已卸载提供商模块: {module_name}")
            except Exception as e:
                print(f"卸载提供商模块 {module_name} 时出错: {str(e)}")
    
    def _reload_provider_module(self, module_name: str) -> None:
        """
        重新加载提供商模块
        
        参数:
            module_name (str): 模块名称
        """
        with self._lock:
            try:
                # 先卸载模块
                self._unload_provider_module(module_name)
                
                # 然后重新加载
                self._load_provider_module(module_name)
                
                print(f"已重新加载提供商模块: {module_name}")
            except Exception as e:
                print(f"重新加载提供商模块 {module_name} 时出错: {str(e)}")
    
    def _start_watcher(self) -> None:
        """
        启动文件监控
        """
        if not self._observer and self._package_dir:
            try:
                # 创建文件系统观察者
                self._observer = Observer(timeout=5)  # 增加超时时间
                handler = ProviderFileHandler(self)
                
                # 降低watchdog相关日志级别
                logging.getLogger('watchdog').setLevel(logging.ERROR)
                
                # 开始监控提供商目录，排除__pycache__和临时文件
                patterns = ["*.py"]  # 仅监控.py文件
                ignore_patterns = ["*.pyc", "*.pyo", "*~", "*.tmp", "*.bak"]
                ignore_directories = True  # 忽略目录变化
                
                self._observer.schedule(
                    handler, 
                    self._package_dir, 
                    recursive=False
                )
                self._observer.start()
                
                print(f"已启动提供商目录监控: {self._package_dir}")
            except Exception as e:
                print(f"启动提供商目录监控时出错: {str(e)}")
    
    def _stop_watcher(self) -> None:
        """
        停止文件监控
        """
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
                self._observer = None
                print("已停止提供商目录监控")
            except Exception as e:
                print(f"停止提供商目录监控时出错: {str(e)}")
    
    def get_provider(self, provider_id: str) -> ModelProvider:
        """
        获取指定ID的提供商
        
        参数:
            provider_id (str): 提供商ID
        
        返回:
            ModelProvider: 提供商实例
            
        抛出:
            HTTPException: 如果提供商不存在
        """
        with self._lock:
            if provider_id not in self._providers:
                raise HTTPException(status_code=404, detail=f"提供商 '{provider_id}' 不存在")
            return self._providers[provider_id]
    
    def get_all_providers(self) -> List[Dict[str, Any]]:
        """
        获取所有可用提供商的信息
        
        返回:
            List[Dict[str, Any]]: 提供商信息列表
        """
        with self._lock:
            return [provider.to_provider_info() for provider in self._providers.values()]
    
    def get_module_details(self) -> Dict[str, Any]:
        """
        获取所有已加载模块的详细信息
        
        返回:
            Dict[str, Any]: 模块和提供商详情
        """
        with self._lock:
            module_details = {
                "loaded_modules": list(self._loaded_modules),
                "total_modules": len(self._loaded_modules),
                "total_providers": len(self._providers),
                "providers_by_module": {}
            }
            
            # 构建模块-提供商映射
            for module_name in self._loaded_modules:
                if module_name in self._provider_classes:
                    module_details["providers_by_module"][module_name] = []
                    
                    for class_name, cls in self._provider_classes[module_name].items():
                        # 找到这个类对应的实例
                        provider_instances = [
                            {
                                "id": p_id,
                                "name": p.provider_name,
                                "description": p.description
                            }
                            for p_id, p in self._providers.items()
                            if p.__class__.__name__ == class_name
                        ]
                        
                        module_details["providers_by_module"][module_name].extend(provider_instances)
            
            return module_details
    
    def reload_providers(self) -> None:
        """
        重新加载所有提供商
        """
        with self._lock:
            # 停止监控
            self._stop_watcher()
            
            # 清除已加载的提供商和模块记录
            self._providers.clear()
            self._loaded_modules.clear()
            self._provider_classes.clear()
            
            # 重新加载所有提供商
            self._load_providers()
            
            # 重启监控
            self._start_watcher()
            
            print("已完成提供商重新加载")
    
    def __del__(self):
        """
        析构函数，确保在对象销毁时停止监控
        """
        self._stop_watcher()


# 创建全局提供商管理器实例
provider_manager = ProviderManager() 
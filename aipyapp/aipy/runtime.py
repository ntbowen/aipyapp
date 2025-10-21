#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from typing import Union, Any, Dict
from functools import wraps
import inspect

from term_image.image import from_file, from_url

from .functions import FunctionManager
from .. import T, TaskPlugin
from ..exec import PythonRuntime
from .blocks import CodeBlock

def restore_output(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

        try:
            return func(self, *args, **kwargs)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    return wrapper

class CliPythonRuntime(PythonRuntime):
    def __init__(self, task):
        super().__init__(envs=task.role.envs, session=task.session)
        self.task = task
        self.gui = task.gui
        self.display = task.display
        self._auto_install = task.settings.get('auto_install')
        self._auto_getenv = task.settings.get('auto_getenv')
        self.function_manager = FunctionManager()

    def register_plugin(self, plugin: TaskPlugin):
        self.function_manager.register_functions(plugin.get_functions())

    @restore_output
    def install_packages(self, *packages: str) -> bool:
        """
        Install third-party packages

        Args:
            packages: The names of the packages to install

        Returns:
            bool: True if the packages are installed successfully, False otherwise

        Examples:
            >>> utils.install_packages('requests', 'openai')
            True
            >>> utils.install_packages('requests', 'openai')
            False
        """
        message = f"LLM {T('Request to install third-party packages')}: {packages}"
        self.task.emit('runtime_message', message=message, status='warning')
        
        if self.display:
            prompt = f"{T('If you agree, please enter')} 'y'> "
            ok = self.display.confirm(prompt, auto=self._auto_install)
        else:
            ok = True
            
        if ok:
            ret = self.ensure_packages(*packages)
            result_message = T("Package installation completed") if ret else T("Package installation failed")
            self.task.emit('runtime_message', message=result_message, status='success' if ret else 'error')
            return ret
        return False
    
    @restore_output
    def get_env(self, name: str, default: str = None, *, desc: str = None) -> Union[str, None]:
        message = f"LLM {T('Request to obtain environment variable {}, purpose', name)}: {desc}"
        self.task.emit('runtime_message', message=message)
        
        try:
            value = self.envs[name][0]
            success_message = f"{T('Environment variable {} exists, returned for code use', name)}"
            self.task.emit('runtime_message', message=success_message)
        except KeyError:
            if self._auto_getenv:
                auto_message = T('Auto confirm')
                self.task.emit('runtime_message', message=auto_message)
                value = None
            elif self.display:
                prompt = f"{T('Environment variable {} not found, please enter', name)}: "
                value = self.display.input(prompt)
                value = value.strip()
            else:
                value = None
            if value:
                self.set_env(name, value, desc)
        return value or default
    
    @restore_output
    def show_image(self, path: str = None, url: str = None) -> None:
        """
        Display an image

        Args:
            path: The path of the image
            url: The URL of the image
        """
        self.task.emit('show_image', path=path, url=url)
        if not self.gui:
            image = from_file(path) if path else from_url(url)
            image.draw()

    @restore_output
    def input(self, prompt: str) -> str:
        self.task.emit('runtime_input', prompt=prompt)
        if self.display:
            return self.display.input(prompt)
        return None
    
    def get_block_by_name(self, block_name: str) -> Union[CodeBlock, None]:
        """
        Get a code block by name

        Args:
            block_name: The name of the code block

        Returns:
            CodeBlock: The code block objector None if not found
        """
        return self.task.code_blocks.get_block_by_name(block_name)
    
    def call_function(self, name: str, **kwargs) -> Any:
        """
        Call a registered function

        Args:
            name: The name of the function to call
            **kwargs: The keyword arguments to pass to the function

        Returns:
            Any: The result of the function call or raise an exception

        Examples:
            >>> utils.call_function('get_env', name='PATH')
            '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'
            >>> utils.call_function('get_env', name='PATH')
            None
        """
        self.task.emit('function_call_started', funcname=name, kwargs=kwargs)
        try:
            result = self.function_manager.call(name, **kwargs)
            self.task.emit('function_call_completed', funcname=name, kwargs=kwargs, result=result, success=True)
            return result
        except Exception as e:
            self.task.emit('function_call_completed', funcname=name, kwargs=kwargs, result=None, success=False, error=str(e), exception=e)
            raise
    
    def get_builtin_functions(self) -> Dict[str, Dict[str, str]]:
        """
        根据函数签名和docstring，生成函数调用提示
        """
        functions = {}
        
        # 内置运行时函数
        builtin_names = ['set_state', 'get_block_state', 'set_persistent_state', 'get_persistent_state', 'install_packages', 'get_env', 'show_image', 'get_block_by_name', 'call_function', 'save_shared_data', 'load_shared_data']
        for name in builtin_names:
            func_obj = getattr(self, name)
            docstring = func_obj.__doc__
            signature = inspect.signature(func_obj)
            functions[name] = {
                'docstring': docstring,
                'signature': signature,
            }
        return functions
    
    def get_plugin_functions(self):
        return self.function_manager.get_functions()
        
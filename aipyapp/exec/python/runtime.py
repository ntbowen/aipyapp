#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import subprocess
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

class PythonRuntime(ABC):
    def __init__(self, envs=None, session=None):
        self.envs = envs if envs is not None else {}
        self.session = session if session is not None else {}
        self.packages = set()
        self.block_states = {}
        self.current_state = {}
        self.block = None
        self.log = logger.bind(src='runtime')

    def start_block(self, block):
        """开始一个新的代码块执行"""
        self.current_state = {}
        self.block_states[block.name] = self.current_state
        self.block = block

    def set_state(self, success: bool, **kwargs) -> None:
        """
        Set the state of the current code block

        Args:
            success: Whether the code block is successful
            **kwargs: Other state values

        Example:
            set_state(success=True, result="Hello, world!")
            set_state(success=False, error="Error message")
        """
        self.current_state['success'] = success
        self.current_state.update(kwargs)

    def get_block_state(self, block_name: str) -> Any:
        """
        Get the state of code block by name

        Args:
            block_name: The name of the code block

        Returns:
            Any: The state of the code block

        Example:
            state = get_block_state("code_block_name")
            if state.get("success"):
                print(state.get("result"))
            else:
                print(state.get("error"))
        """
        return self.block_states.get(block_name)
    
    def set_persistent_state(self, **kwargs) -> None:
        """
        Set the state of the current code block in the session

        Args:
            **kwargs: The state values
        """
        self.session.update(kwargs)
        self.block.add_dep('set_state', list(kwargs.keys()))

    def get_persistent_state(self, key: str) -> Any:
        """
        Get the state of the current code block in the session

        Args:
            key: The key of the state

        Returns:
            Any: The state of the code block
        """
        self.block.add_dep('get_state', key)
        return self.session.get(key)

    def set_env(self, name, value, desc):
        self.envs[name] = (value, desc)

    def ensure_packages(self, *packages, upgrade=False, quiet=False):
        if not packages:
            return True

        packages = list(set(packages) - self.packages)
        if not packages:
            return True
        
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        if quiet:
            cmd.append("-q")
        cmd.extend(packages)

        try:
            subprocess.check_call(cmd)
            self.packages.update(packages)
            return True
        except subprocess.CalledProcessError:
            self.log.error("依赖安装失败: {}", " ".join(packages))
        
        return False

    def ensure_requirements(self, path="requirements.txt", **kwargs):
        with open(path) as f:
            reqs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return self.ensure_packages(*reqs, **kwargs)
    
    @abstractmethod
    def install_packages(self, *packages: str):
        pass

    @abstractmethod
    def get_env(self, name: str, default: Any = None, *, desc: str = None) -> Any:
        pass
    
    @abstractmethod
    def show_image(self, path: str = None, url: str = None) -> None:
        pass

    @abstractmethod
    def input(self, prompt: str = '') -> str:
        pass
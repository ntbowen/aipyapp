#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
from pathlib import Path
from typing import Union
from functools import wraps
from importlib.resources import read_text

from rich.panel import Panel

from .. import T, __respkg__

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

def confirm_disclaimer(console):
    DISCLAIMER_TEXT = read_text(__respkg__, "DISCLAIMER.md")
    console.print()
    panel = Panel.fit(DISCLAIMER_TEXT, title="[red]免责声明", border_style="red", padding=(1, 2))
    console.print(panel)

    while True:
        console.print("\n[red]是否确认已阅读并接受以上免责声明？[/red](yes/no):", end=" ")
        response = input().strip().lower()
        if response in ("yes", "y"):
            console.print("[green]感谢确认，程序继续运行。[/green]")
            return True
        elif response in ("no", "n"):
            console.print("[red]您未接受免责声明，程序将退出。[/red]")
            return False
        else:
            console.print("[yellow]请输入 yes 或 no。[/yellow]")

def safe_rename(path: Path, input_str: str, max_length=16) -> Path:
    input_str = input_str.strip()
    safe_str = re.sub(r'[\\/:*?"<>|\s]', '', input_str).strip()
    if not safe_str:
        safe_str = "Task"

    name = safe_str[:max_length]
    new_path = path.parent / f"{name}{path.suffix}"
    counter = 1

    while True:
        if not new_path.exists():
            try:
                path.rename(new_path)
                break
            except FileExistsError:
                pass
            except OSError as e:
                if e.errno in (os.errno.EEXIST, os.errno.ENOTEMPTY):
                    pass
                else:
                    raise
        new_path = path.parent / f"{name}_{counter}{path.suffix}"
        counter += 1

    return new_path

def validate_file(path: Union[str, Path]) -> None:
    """验证文件格式和存在性"""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    
    if not path.name.endswith('.json'):
        raise ValueError("Task file must be a .json file")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

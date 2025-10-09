#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from importlib.resources import read_text

from rich.console import Console
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from ..aipy import TaskManager
from .. import T, __version__, __respath__
from .command import CommandManager, TaskModeResult, CommandError, CommandResult, CommandManagerConfig, CommandContext
from ..display import DisplayManager

STYLE_MAIN = {
    'completion-menu.completion': 'bg:green #ffffff',
    'completion-menu.completion.current': 'bg:#444444 #ffffff',
    'completion-menu.meta': 'bg:#000000 #999999',
    'completion-menu.meta.current': 'bg:#444444 #aaaaaa',
    'prompt': 'green',
    'bottom-toolbar': 'bg:#FFFFFF green'
}

STYLE_TASK = {
    'completion-menu.completion': 'bg:#008080 #ffffff',         # 深蓝背景，白色文本
    'completion-menu.completion.current': 'bg:#005577 #ffffff', # 当前选中，亮蓝
    'completion-menu.meta': 'bg:#002244 #cccccc',               # 补全项的 meta 信息
    'completion-menu.meta.current': 'bg:#005577 #eeeeee',       # 当前选中的 meta
    'prompt': '#008080',
    'bottom-toolbar': "bg:#880000 #008080"
}

class InteractiveConsole():
    def __init__(self, task_manager, console, settings):
        """
        初始化控制台
        
        Args:
            settings: 应用设置
            task_manager: TaskManager 实例
            console: Rich Console 实例
        """
        self.settings = settings
        self.tm = task_manager
        self.console = console
        self.task = None

        # 创建历史记录
        self.history = FileHistory(str(settings['config_dir'] / ".history"))
        
        # 创建命令管理器配置
        command_config = self._create_command_config()
        
        # 创建运行时上下文
        self.command_context = self._create_command_context()
        
        # 创建命令管理器
        self.command_manager = CommandManager(command_config, self.command_context)
        
        # 创建提示会话
        self.session = self._create_prompt_session()
        
        # 样式
        self.style_main = Style.from_dict(STYLE_MAIN)
        self.style_task = Style.from_dict(STYLE_TASK)

    def _create_command_config(self) -> CommandManagerConfig:
        """创建命令管理器配置"""
        return CommandManagerConfig(
            settings=self.settings,
            builtin_command_dir=Path(__respath__ / "commands"),
            custom_command_dirs=[
                Path(self.settings['config_dir']) / "commands",
                # 可以添加更多自定义命令目录
            ]
        )
    
    def _create_command_context(self) -> CommandContext:
        """创建运行时上下文"""
        return CommandContext(
            tm=self.tm,
            task=None,
            console=self.console,
            settings=self.settings
        )

    def _create_prompt_session(self) -> PromptSession:
        """创建提示会话"""
        return PromptSession(
            history=self.history,
            completer=self.command_manager,  # CommandManager 实现了 Completer 接口
            auto_suggest=AutoSuggestFromHistory(),
            bottom_toolbar=self.get_bottom_toolbar,
            key_bindings=self.command_manager.key_bindings
        )
           
    def get_main_status(self):
        status = self.tm.get_status()
        try:
            mcp_text = f" | MCP: {T('Enabled') if status['mcp_enabled'] else T('Disabled')}"
        except KeyError:
            mcp_text = ""
        return f"LLM: {status['llm']} | Role: {status['role']} | Display: {status['display']} | Tasks: {status['tasks']}{mcp_text}"
    
    def get_task_status(self):
        if self.command_context.task:
            status = self.command_context.task.get_status()
            return f"LLM: {status['llm']} | Blocks: {status['blocks']} | Steps: {status['steps']}"
        return ""
    
    def get_bottom_toolbar(self):
        if self.command_context.is_task_mode():
            status = self.get_task_status()
            text = f"[AI] {status}"
        else:
            status = self.get_main_status()
            text = f"[Main] {status}"
        return [('class:bottom-toolbar', text)]
    
    def _input_with_multiline(self, prompt_text, task_mode=False):
        """获取用户输入（支持多行）"""
        style = self.style_task if task_mode else self.style_main
        cursor_shape = CursorShape.BLOCK if task_mode else CursorShape.BEAM
        
        first_line = self.session.prompt(
            [("class:prompt", prompt_text)],
            style=style,
            cursor=cursor_shape
        )
        
        if not first_line.endswith("\\"):
            return first_line
        
        # 多行输入
        lines = [first_line.rstrip("\\")]
        while True:
            next_line = self.session.prompt(
                [("class:prompt", "... ")],
                style=style
            )
            if next_line.endswith("\\"):
                lines.append(next_line.rstrip("\\"))
            else:
                lines.append(next_line)
                break
        
        return "\n".join(lines)

    def run_task(self, task, instruction, title=None):
        try:
            if instruction[0] == '|':
                task.run_subtask(instruction[1:].rstrip(), title=title)
            else:
                task.run(instruction, title=title)
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            self.console.print_exception()

    def start_task_mode(self, task, instruction=None, title=None):
        self.command_context.set_task_mode(task)
        if instruction:
            self.console.print(f"[AI] {T('Enter Ctrl+d or /done to end current task')}", style="dim color(240)")
            self.run_task(task, instruction, title=title)
        else:
            self.console.print(f"[AI] {T('Resuming task')}: {task.task_id}", style="dim color(240)")
            
        while True:
            try:
                user_input = self._input_with_multiline(">>> ", task_mode=True).strip()
                if len(user_input) < 2: continue
            except (EOFError, KeyboardInterrupt):
                break

            if user_input in ('/done', 'done'):
                break

            if not user_input.startswith('/'):
                self.run_task(task, user_input)
                continue

            try:
                self.command_manager.execute(user_input)
            except CommandError as e:
                self.console.print(f"[red]{e}[/red]")

        try:
            task.done()
        except Exception as e:
            self.console.print_exception()
        self.console.print(f"[{T('Exit AI mode')}]", style="dim")

    def run(self):
        self.console.print(f"[Main] {T('Please enter an instruction or `/help` for more information')}", style="dim color(240)")
        tm = self.tm
        while True:
            self.command_context.set_main_mode()
            try:
                user_input = self._input_with_multiline(">> ").strip()
                if len(user_input) < 2:
                    continue

                if not user_input.startswith('/'):
                    task = tm.new_task()
                    self.start_task_mode(task, user_input)
                    continue

                try:
                    ret = self.command_manager.execute(user_input)
                    if isinstance(ret, CommandResult) and isinstance(ret.result, TaskModeResult):
                        result = ret.result
                        if result.instruction:
                            task = tm.new_task()
                            title = result.title
                        else:
                            task = result.task
                            title = None
                        self.start_task_mode(task, result.instruction, title=title)
                        continue
                except CommandError as e:
                    self.console.print(f"[red]{e}[/red]")
            except (EOFError, KeyboardInterrupt):
                break

def get_logo_text(config_dir):
    path = config_dir / "logo.txt"
    if path.exists():
        logo_text = path.read_text()
    else:
        logo_text = Path(__respath__ / "logo.txt").read_text()
    return logo_text

def main(settings):
    console = Console(record=True)
    console.print(f"🚀 Python use - AIPython ({__version__}) [[pink]https://aipy.app[/pink]]", style="bold green")
    logo_text = get_logo_text(settings['config_dir'])
    console.print(Text.from_ansi(logo_text))

    # 初始化显示效果管理器
    display_config = settings.get('display', {})
    display_manager = DisplayManager(display_config, console=console)
    try:
        tm = TaskManager(settings, display_manager=display_manager)
    except Exception as e:
        console.print_exception()
        return
    
    update = tm.get_update()
    if update and update.get('has_update'):
        console.print(f"[bold red]🔔 号外❗ {T('Update available')}: {update.get('latest_version')}")
   
    if not tm.client_manager:
        console.print(f"[bold red]{T('No available LLM, please check the configuration file')}")
        return
    
    cmd = settings.get('exec_cmd')
    run_json = settings.get('run_json')
    if cmd or run_json:
        if run_json:
            task = tm.load_task(run_json)
        else:
            task = tm.new_task()

        cmd = cmd or 'Run the task again'
        task.run(cmd)
        task.done()
        return
    
    InteractiveConsole(tm, console, settings).run()

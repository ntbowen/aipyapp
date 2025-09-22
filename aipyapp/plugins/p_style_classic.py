#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from functools import wraps
import json

from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.tree import Tree
from rich.text import Text
from rich.console import Console

from aipyapp.display import RichDisplayPlugin
from live_display import LiveDisplay
from aipyapp import T

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

class DisplayClassic(RichDisplayPlugin):
    """Classic display style"""
    name = "classic"
    version = "1.0.0"
    description = "Classic display style"
    author = "AiPy Team"

    def __init__(self, console: Console, quiet: bool = False):
        super().__init__(console, quiet)
        self.live_display = None

    def _get_title(self, title: str, *args, style: str = "info", prefix: str = "\n"):
        text = Text(f"{prefix}● {title}".format(*args), style=style)
        text.highlight_words(args, style="bold white")
        return text

    def on_exception(self, event):
        """异常事件处理"""
        msg = event.typed_event.msg
        exception = event.typed_event.exception
        title = self._get_title(T("Exception occurred"), msg, style="error")
        tree = Tree(title)
        tree.add(exception)
        self.console.print(tree)

    def on_task_started(self, event):
        """任务开始事件处理"""
        instruction = event.typed_event.instruction
        title = event.typed_event.title
        if not title:
            title = instruction
        tree = Tree(f"🚀 {T('Task processing started')}")
        tree.add(title)
        self.console.print(tree)

    def on_request_started(self, event):
        """查询开始事件处理"""
        llm = event.typed_event.llm
        title = self._get_title(T("Sending message to {}"), llm)
        self.console.print(title)

    def on_step_started(self, event):
        """步骤开始事件处理"""
        instruction = event.typed_event.instruction
        title = event.typed_event.title
        if not title:
            title = instruction
        prompt = self._get_title(T("Instruction processing started"))
        tree = Tree(prompt)
        tree.add(title)
        self.console.print(tree)

    def on_stream_started(self, event):
        """流式开始事件处理"""
        if not self.quiet:
            self.live_display = LiveDisplay()
            self.live_display.__enter__()
            title = self._get_title(T("Streaming started"), prefix="")
            self.console.print(title)
    
    def on_stream_completed(self, event):
        """流式结束事件处理"""
        if self.live_display:
            self.live_display.__exit__(None, None, None)
            self.live_display = None

    def on_stream(self, event):
        """LLM 流式响应事件处理"""
        lines = event.typed_event.lines
        reason = event.typed_event.reason
        if self.live_display:
            self.live_display.update_display(lines, reason=reason)

    @staticmethod
    def convert_front_matter(md_text: str) -> str:
        pattern = r"^---\s*\n(.*?)\n---\s*\n"
        #return re.sub(pattern, r"```yaml\n\1\n```\n", md_text, flags=re.DOTALL)
        return re.sub(pattern, "", md_text, flags=re.DOTALL)
          
    def on_response_completed(self, event):
        """LLM 响应完成事件处理"""
        llm = event.typed_event.llm
        msg = event.typed_event.msg
        if not msg:
            title = self._get_title(T("LLM response is empty"), style="error")
            self.console.print(title)
            return
        
        if msg.role == 'error':
            title = self._get_title(T("Failed to receive message"), style="error")
            tree = Tree(title)
            tree.add(msg.content)
            self.console.print(tree)
            return
        
        content = self.convert_front_matter(msg.content)
        if msg.reason:
            content = f"{msg.reason}\n\n-----\n\n{content}"
        title = self._get_title(f"{T('Completed receiving message')} ({llm})", style="success")
        tree = Tree(title)
        tree.add(Markdown(content))
        self.console.print(tree)

    def on_task_status(self, event):
        """任务状态事件处理"""
        status = event.typed_event.status
        completed = status.completed
        style = "success" if completed else "error" 
        title = self._get_title(T("Task status"), style=style)
        tree = Tree(title, guide_style=style)
        if completed:
            tree.add(T("Completed"))
            tree.add(T("Confidence level: {}", status.confidence))
        else:
            tree.add(status.status)
            if status.reason:
                tree.add(T("Reason: {}", status.reason))
            if status.suggestion:
                tree.add(T("Suggestion: {}", status.suggestion))
        self.console.print(tree)

    def on_parse_reply_completed(self, event):
        """消息解析结果事件处理"""
        response = event.typed_event.response
        if not response:
            return
            
        title = self._get_title(T("Message parse result"))
        tree = Tree(title)
        
        if response.code_blocks:
            block_names = [f"{block.name}/{block.lang}" for block in response.code_blocks]    
            block_str = ", ".join(block_names[:3])
            if len(block_names) > 3:
                block_str += f" (+{len(block_names)-3} more)"
            tree.add(f"{T('Blocks')}: {block_str}")
        
        if response.tool_calls:
            sub_tree = tree.add(T('Tool Calls'))
            for tool_call in response.tool_calls:
                if tool_call.name == 'Exec':    
                    sub_tree.add(f"{T('Exec')}: {tool_call.arguments.name}")
                elif tool_call.name == 'Edit':
                    sub_tree.add(f"{T('Edit')}: {tool_call.arguments.name}")
                else:
                    sub_tree.add(f"{tool_call.name.value}: {tool_call.arguments}")
            
        errors = response.errors
        if errors:
            et = tree.add(T('Errors'))
            for error in errors:
                et.add(error.message)
        
        self.console.print(tree)

    def on_exec_started(self, event):
        """代码执行开始事件处理"""
        block = event.typed_event.block
        title = self._get_title(T("Start executing code block {}"), block.name)
        self.console.print(title)
        
    def on_edit_started(self, event):
        """代码编辑开始事件处理"""
        block = event.typed_event.block
        old_str = block.old
        new_str = block.new
        
        title = self._get_title(T("Start editing code block {}"), block.name, style="warning")
        tree = Tree(title)
        
        if old_str:
            old_preview = old_str[:50] + '...' if len(old_str) > 50 else old_str
            tree.add(f"{T('Replace')}: {repr(old_preview)}")
        if new_str:
            new_preview = new_str[:50] + '...' if len(new_str) > 50 else new_str
            tree.add(f"{T('With')}: {repr(new_preview)}")
            
        self.console.print(tree)
        
    def on_edit_completed(self, event):
        """代码编辑结果事件处理"""
        typed_event = event.typed_event
        success = typed_event.success
        new_version = typed_event.new_version
        block_name = typed_event.block_name
        
        if success:
            style = "success"
            title = self._get_title(T("Edit completed {}"), block_name, style=style)
            tree = Tree(title)
            
            if new_version:
                tree.add(f"{T('New version')}: v{new_version}")
        else:
            style = "error"
            title = self._get_title(T("Edit failed {}"), block_name, style=style)
            tree = Tree(title)
            tree.add(T("Edit operation failed"))
            
        self.console.print(tree)
            
    @restore_output
    def on_function_call_started(self, event):
        """函数调用事件处理"""
        funcname = event.typed_event.funcname
        kwargs = event.typed_event.kwargs
        title = self._get_title(T("Start calling function {}"), funcname)
        tree = Tree(title)
        json_kwargs = json.dumps(kwargs, ensure_ascii=False, default=str)
        tree.add(json_kwargs[:64] + '...' if len(json_kwargs) > 64 else json_kwargs)
        self.console.print(tree)

    @restore_output
    def on_function_call_completed(self, event):
        """函数调用结果事件处理"""
        funcname = event.typed_event.funcname
        success = event.typed_event.success
        result = event.typed_event.result
        error = event.typed_event.error
        
        if success:
            style = "success"
            title = self._get_title(T("Function call result {}"), funcname, style=style)
            tree = Tree(title)
            if result is not None:
                # 格式化并显示结果
                if isinstance(result, (dict, list)):
                    json_result = json.dumps(result, ensure_ascii=False, indent=2, default=str)
                    tree.add(Syntax(json_result, "json", word_wrap=True, line_range=(0, 10)))
                else:
                    tree.add(str(result))
            else:
                tree.add(T("No return value"))
            self.console.print(tree)
        else:
            style = "error"
            title = self._get_title(T("Function call failed {}"), funcname, style=style)
            tree = Tree(title)
            tree.add(error if error else T("Unknown error"))
            self.console.print(tree)

    def on_exec_completed(self, event):
        """代码执行结果事件处理"""
        typed_event = event.typed_event
        result = typed_event.result
        block = typed_event.block
        
        try:
            style = "error" if result.has_error() else "success"
        except:
            style = "warning"
        
        # 显示说明信息
        title = self._get_title(T("Execution result {}"), block.name, style=style)
        tree = Tree(title)
        
        # JSON格式化和高亮显示结果
        json_result = result.model_dump_json(indent=2, exclude_none=True)
        tree.add(Syntax(json_result, "json", word_wrap=True))
        self.console.print(tree)

    def on_tool_call_started(self, event):
        """工具调用开始事件处理"""
        tool_call = event.typed_event.tool_call
        title = self._get_title(T("Start calling tool {}"), tool_call.name.value)
        tree = Tree(title)
        args = tool_call.arguments.model_dump_json()
        tree.add(args[:64] + '...' if len(args) > 64 else args)
        self.console.print(tree)

    def on_tool_call_completed(self, event):
        """MCP 工具调用结果事件处理"""
        typed_event = event.typed_event
        result = typed_event.result
        title = self._get_title(T("Tool call result {}"), result.tool_name.value)
        tree = Tree(title)
        json_result = result.result.model_dump_json(indent=2, exclude_none=True)
        tree.add(Syntax(json_result, "json", word_wrap=True))
        self.console.print(tree)

    def on_step_completed(self, event):
        """任务总结事件处理"""
        summary = event.typed_event.summary
        usages = summary.get('usages', [])
        if usages:
            table = Table(title=T("Task Summary"), show_lines=True)

            table.add_column(T("Round"), justify="center", style="bold cyan", no_wrap=True)
            table.add_column(T("Time(s)"), justify="right")
            table.add_column(T("In Tokens"), justify="right")
            table.add_column(T("Out Tokens"), justify="right")
            table.add_column(T("Total Tokens"), justify="right", style="bold magenta")

            round = 1
            for row in usages:
                table.add_row(
                    str(round),
                    str(row["time"]),
                    str(row["input_tokens"]),
                    str(row["output_tokens"]),
                    str(row["total_tokens"]),
                )
                round += 1
            self.console.print("\n")
            self.console.print(table)

        summary = summary.get('summary')
        title = self._get_title(T("End processing instruction"))
        tree = Tree(title)
        tree.add(f"{T('Summary')}: {summary}")
        self.console.print(tree)

    def on_step_cleanup_completed(self, event):
        """Step清理完成事件处理"""
        typed_event = event.typed_event
        cleaned_messages = typed_event.cleaned_messages
        remaining_messages = typed_event.remaining_messages
        tokens_saved = typed_event.tokens_saved
        tokens_remaining = typed_event.tokens_remaining
        
        title = self._get_title(T("Context cleanup completed"), style="dim cyan")
        tree = Tree(title)
        tree.add(T("🧹 Cleaned {} messages", cleaned_messages))
        tree.add(T("📝 {} messages remaining", remaining_messages))
        tree.add(T("🔥 Saved {} tokens", tokens_saved))
        tree.add(T("📊 {} tokens remaining", tokens_remaining))
        tree.add(T("📉 Context optimized for better performance"))
        self.console.print(tree)

    def on_upload_result(self, event):
        """云端上传结果事件处理"""
        status_code = event.typed_event.status_code
        url = event.typed_event.url
        if url:
            self.console.print(f"🟢 {T('Article uploaded successfully, {}', url)}", style="success")
        else:
            self.console.print(f"🔴 {T('Upload failed (status code: {})', status_code)}", style="error")

    def on_task_completed(self, event):
        """任务结束事件处理"""
        path = event.typed_event.path
        title = self._get_title(T("Task completed"))
        tree = Tree(title)
        if path:
            tree.add(f"{T('Path')}: {path}")
        self.console.print(tree)

    def on_runtime_message(self, event):
        """Runtime消息事件处理"""
        message = event.typed_event.message
        status = event.typed_event.status or 'info'
        title = self._get_title(message, style=status)
        self.console.print(title)

    def on_runtime_input(self, event):
        """Runtime输入事件处理"""
        # 输入事件通常不需要特殊处理，因为input_prompt已经处理了
        pass
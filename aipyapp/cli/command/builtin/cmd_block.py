#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional
from rich.syntax import Syntax

from .utils import row2table
from ..base import CommandMode, ParserCommand
from aipyapp import T

class BlockCommand(ParserCommand):
    """Block command"""
    name = "block"
    description = T("Manage code blocks")
    modes = [CommandMode.TASK]
    
    def get_arg_values(self, name, subcommand=None, partial=None) -> Optional[List[Tuple[str, str]]]:
        if name == 'name':
            ctx = self.manager.context
            return [(block.name, block.lang) for block in ctx.task.blocks]
        return None

    def add_subcommands(self, subparsers):
        subparsers.add_parser('list', help=T('List code blocks'))
        parser = subparsers.add_parser('show', help=T('Show code block source'))
        parser.add_argument('name', type=str, help=T('Name of the code block'))
        parser = subparsers.add_parser('run', help=T('Run code block'))
        parser.add_argument('name', type=str, help=T('Name of the code block'))

    def cmd(self, args, ctx):
        return self.cmd_list(args, ctx)
    
    def cmd_show(self, args, ctx):
        """显示代码块源码"""
        task = ctx.task
        block = task.blocks.get(args.name)
        if not block:
            ctx.console.print(T("Code block not found"))
            return False
        
        syntax = Syntax(block.code, block.lang, theme="github-dark", line_numbers=False)
        
        ctx.console.print(f"\n[bold]{T('Code Block')}: {block.name} (v{block.version})[/bold]")
        if block.path:
            ctx.console.print(f"[dim]{T('Path')}: {block.path}[/dim]")
        ctx.console.print("")
        ctx.console.print(syntax)
        return True
    
    def cmd_run(self, args, ctx):
        """运行代码块"""
        task = ctx.task
        return task.run_block(args.name)
    
    def cmd_list(self, args, ctx):
        """列出所有代码块"""
        task = ctx.task
        rows = []
        for block in task.blocks:
            # 计算代码大小
            code_size = len(block.code) if block.code else 0
            size_str = f"{code_size} chars"
            
            # 处理路径显示
            path_str = block.path if block.path else '-'
            if path_str != '-' and len(path_str) > 33:
                path_str = '...' + path_str[-30:]
            
            rows.append([
                block.name,
                f"v{block.version}",
                block.lang,
                path_str,
                size_str
            ])
        table = row2table(rows, title=T("Code Blocks"), headers=[T('Name'), T('Version'), T('Language'), T('Path'), T('Size')])
        ctx.console.print(table)
        return True
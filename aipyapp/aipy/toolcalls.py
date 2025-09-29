#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""工具调用系统模块"""

from __future__ import annotations
from enum import Enum
from typing import Union, List, Dict, Any, Optional, TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, model_validator, Field

from .types import Error
from ..exec import ExecResult, ProcessResult, PythonResult

if TYPE_CHECKING:
    from .task import Task

class ToolName(str, Enum):
    """Tool name"""
    EDIT = "Edit"
    EXEC = "Exec"
    MCP = "MCP"
    SUBTASK = "SubTask"

class ToolResult(BaseModel):
    """Tool result"""
    error: Error | None = Field(title="Tool error", default=None)

    def to_json(self):
        return self.model_dump_json(exclude_none=True, exclude_unset=True)

class ExecToolArgs(BaseModel):
    """Exec tool arguments"""
    name: str = Field(title="Code block name to execute", min_length=1, strip_whitespace=True)

class ExecToolResult(ToolResult):
    """Exec tool result"""
    block_name: str = Field(title="Code block name executed", min_length=1, strip_whitespace=True)
    result: ExecResult | ProcessResult | PythonResult | None = Field(title="Execution result", default=None)

class EditToolArgs(BaseModel):
    """Edit tool arguments"""
    name: str = Field(title="Code block name to edit", min_length=1, strip_whitespace=True)
    old: str = Field(title="Code to replace", min_length=1)
    new: str = Field(title="Replacement code")
    replace_all: Optional[bool] = Field(False, title="Replace all occurrences")

class EditToolResult(ToolResult):
    """Edit tool result"""
    block_name: str = Field(title="Code block name edited", min_length=1, strip_whitespace=True)
    new_version: int | None = Field(title="New version number", gt=1, default=None)

class MCPToolArgs(BaseModel):
    """MCP tool arguments"""
    model_config = {
        'extra': 'allow'
    }

class MCPToolResult(ToolResult):
    """MCP tool result"""
    result: Dict[str, Any] = Field(default_factory=dict)

class SubTaskArgs(BaseModel):
    """SubTask tool arguments"""
    instruction: str = Field(title="SubTask instruction", min_length=1, strip_whitespace=True)
    title: Optional[str] = Field(default=None, title="SubTask title")
    inherit_context: bool = Field(default=False, title="Whether to inherit parent context")

class SubTaskResult(ToolResult):
    """SubTask tool result"""
    subtask_id: str = Field(title="SubTask ID")
    status: str = Field(title="SubTask status")  # 'completed', 'failed', 'timeout'
    result: Optional[str] = Field(default=None, title="SubTask result content")
    execution_time: float = Field(title="Execution time in seconds")
    steps_count: int = Field(title="Number of steps executed")

class ToolCall(BaseModel):
    """Tool call"""
    name: ToolName
    arguments: Union[ExecToolArgs, EditToolArgs, MCPToolArgs, SubTaskArgs]

    @model_validator(mode='before')
    @classmethod
    def alias_name(cls, values: Dict[str, Any]):
        if isinstance(values, dict):
            if "name" not in values and "action" in values:
                values["name"] = values.pop("action")
        return values
       
    def __str__(self):
        return f"ToolCall(name='{self.name}', args={self.arguments})"
    
    def __repr__(self):
        return self.__str__()

class ToolCallResult(BaseModel):
    """Tool call result"""
    tool_name: ToolName
    result: Union[ExecToolResult, EditToolResult, MCPToolResult, SubTaskResult] = Field(title="Tool result")

class ToolCallProcessor:
    """工具调用处理器 - 高级接口"""
    
    def __init__(self):
        self.log = logger.bind(src='ToolCallProcessor')
    
    def process(self, task: 'Task', tool_calls: List[ToolCall]) -> List[ToolCallResult]:
        """
        处理工具调用列表
        
        Args:
            tool_calls: ToolCall 对象列表
            
        Returns:
            List[ToolCallResult]: 包含所有执行结果的列表
        """
        results = []
        failed_blocks = set()  # 记录编辑失败的代码块
        
        for tool_call in tool_calls:
            name = tool_call.name
            if name == ToolName.EXEC:
                # 如果这个代码块之前编辑失败，跳过执行
                block_name = tool_call.arguments.name
                if block_name in failed_blocks:
                    error = Error.new(
                        'Execution skipped: previous edit of the block failed',
                        block_name=block_name
                    )
                    results.append(ToolCallResult(
                        tool_name=name,
                        result=ExecToolResult(
                            block_name=block_name,
                            error=error
                        )
                    ))
                    continue
            
            # 执行工具调用
            result = self.call_tool(task, tool_call)
            results.append(result)
            
            if name == ToolName.EDIT and result.result.error:
                failed_blocks.add(tool_call.arguments.name)
        
        return results

    def call_tool(self, task: 'Task', tool_call: ToolCall) -> ToolCallResult:
        """
        执行工具调用
        
        Args:
            tool_call: ToolCall 对象
            
        Returns:
            ToolResult: 执行结果
        """
        task.emit('tool_call_started', tool_call=tool_call)
        if tool_call.name == ToolName.EXEC:
            result = self._call_exec(task, tool_call)
        elif tool_call.name == ToolName.EDIT:
            result = self._call_edit(task, tool_call)
        elif tool_call.name == ToolName.MCP:
            result = self._call_mcp(task, tool_call)
        elif tool_call.name == ToolName.SUBTASK:
            result = self._call_subtask(task, tool_call)
        else:
            result = ToolResult(error=Error('Unknown tool'))

        toolcall_result = ToolCallResult(
            tool_name=tool_call.name,
            result=result
        )
        task.emit('tool_call_completed', result=toolcall_result)
        return toolcall_result
           
    def _call_edit(self, task: 'Task', tool_call: ToolCall) -> EditToolResult:
        """执行 Edit 工具"""
        args = tool_call.arguments
        block_name = args.name

        original_block = task.blocks.get(block_name)
        if not original_block:
            return EditToolResult(block_name=block_name, error=Error.new("Code block not found"))
        
        old_str = args.old
        new_str = args.new
        replace_all = args.replace_all

        # 检查是否找到匹配的字符串
        if old_str not in original_block.code:
            return EditToolResult(block_name=block_name, error=Error.new(f"No match found for {old_str[:50]}..."))
        
        # 检查匹配次数
        match_count = original_block.code.count(old_str)
        if match_count > 1 and not replace_all:
            return EditToolResult(block_name=block_name, error=Error.new(f"Multiple matches found for {old_str[:50]}...", suggestion="set replace_all: true or provide more specific context"))
        
        # 执行替换生成新代码
        new_code = original_block.code.replace(old_str, new_str, -1 if replace_all else 1)
        
        # 创建新的代码块（版本号+1）
        new_block = original_block.model_copy(
            update={
                "version": original_block.version + 1,
                "code": new_code,
                "deps": original_block.deps.copy() if original_block.deps else {}
            }
        )
        task.blocks.add_block(new_block, validate=False)
        return EditToolResult(block_name=block_name, new_version=new_block.version)
    
    def _call_exec(self, task: 'Task', tool_call: ToolCall) -> ExecToolResult:
        """执行 Exec 工具"""
        args = tool_call.arguments
        block_name = args.name
        
        # 获取代码块
        block = task.blocks.get(block_name)
        if not block:
            return ExecToolResult(
                block_name=block_name,
                error=Error.new("Code block not found")
            )
        
        # 执行代码块
        try:
            result = task.runner(block)
            return ExecToolResult(
                block_name=block_name,
                result=result
            )
        except Exception as e:
            self.log.exception(f"Execution failed with exception: {e}")
            return ExecToolResult(
                block_name=block_name,
                error=Error.new("Execution failed with exception", exception=str(e))
            )
    
    def _call_mcp(self, task: 'Task', tool_call: ToolCall) -> MCPToolResult:
        """执行 MCP 工具"""
        result = task.mcp.call_tool(tool_call.name, tool_call.arguments)
        return MCPToolResult(
            result=result
        )

    def _call_subtask(self, task: 'Task', tool_call: ToolCall) -> SubTaskResult:
        """执行 SubTask 工具"""
        from .subtask import SubTaskManager

        args = tool_call.arguments

        try:
            # 确保任务有SubTaskManager
            if not hasattr(task, 'subtask_manager') or task.subtask_manager is None:
                task.subtask_manager = SubTaskManager(task)

            # 创建子任务
            subtask = task.subtask_manager.create_subtask(
                instruction=args.instruction,
                title=args.title,
                inherit_context=args.inherit_context
            )

            # 运行子任务
            result_dict = task.subtask_manager.run_subtask(
                subtask=subtask,
                instruction=args.instruction,
                title=args.title
            )

            return SubTaskResult(
                subtask_id=result_dict["subtask_id"],
                status=result_dict["status"],
                result=result_dict["result"],
                execution_time=result_dict["execution_time"],
                steps_count=result_dict["steps_count"]
            )

        except Exception as e:
            self.log.exception("SubTask execution failed")
            return SubTaskResult(
                subtask_id="unknown",
                status="failed",
                result=f"SubTask execution failed: {str(e)}",
                execution_time=0.0,
                steps_count=0,
                error=Error.new("SubTask execution failed", exception=str(e))
            )
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, TYPE_CHECKING, Any
import time
from collections import Counter

from loguru import logger
from pydantic import BaseModel, Field

from ..llm import ErrorMessage, UserMessage
from .chat import ChatMessage
from .response import Response
from .toolcalls import ToolCallResult, ToolName
from .prompts import Prompts
from .events import BaseEvent
from .types import DataMixin

if TYPE_CHECKING:
    from .task import Task

class Round(BaseModel):
    # LLM的回复消息
    llm_response: Response = Field(default_factory=Response)
    # 工具调用执行结果
    toolcall_results: List[ToolCallResult] | None = None
    # 系统对执行结果的回应消息(如果有)
    system_feedback: ChatMessage | None = None
    # 上下文清理标记：是否已从上下文中删除
    context_deleted: bool = Field(default=False, description="Whether this round's messages have been deleted from context")

    def should_continue(self) -> bool:
        return self.llm_response.should_continue()
    
    def get_system_feedback(self, prompts: Prompts) -> UserMessage | None:
        if self.llm_response.errors:
            prompt = prompts.get_parse_error_prompt(self.llm_response.errors)
        elif self.toolcall_results:
            prompt = prompts.get_toolcall_results_prompt(self.toolcall_results)
        else:
            return None
        return UserMessage(content=prompt)
    
    def can_safely_delete(self) -> bool:
        """判断Round对应的上下文消息是否可以安全删除
        
        可以安全删除的情况：
        1. LLM回复有解析错误
        2. 所有工具调用都失败
        
        保留的情况：
        3. 纯文本Round（Step自然结束）
        4. 有任何成功的工具调用
        """
        # 1. LLM回复有解析错误 -> 可以删除
        if self.llm_response.errors:
            return True
        
        # 2. 所有工具调用都失败 -> 可以删除
        if self.toolcall_results and all(self._tool_call_failed(tcr) for tcr in self.toolcall_results):
            return True
        
        # 3. 其他情况 -> 保留
        # 包括：纯文本Round（Step结束）和有成功工具调用的Round
        return False
    
    def _tool_call_failed(self, tool_call_result: ToolCallResult) -> bool:
        """判断工具调用是否失败"""
        # 检查工具调用层面的错误
        if tool_call_result.result.error is not None:
            return True
        
        # 对于 Exec 工具，还需要检查实际执行结果
        if tool_call_result.tool_name == ToolName.EXEC:
            exec_result = tool_call_result.result.result
            return exec_result.has_error()
        
        return False
    
class StepData(BaseModel):
    # 用户的初始指令作为Step级别的字段
    initial_instruction: ChatMessage
    instruction: str  # 保持向后兼容
    title: str | None = None
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    
    # 每个Round包含完整的对话+执行循环  
    rounds: List[Round] = Field(default_factory=list)
    
    events: List[BaseEvent.get_subclasses_union()] = Field(default_factory=list)
    
    @property
    def final_response(self):
        return self.rounds[-1].llm_response if self.rounds else None
    
    def add_round(self, round: Round):
        self.rounds.append(round)

class Step:
    def __init__(self, task: Task, data: StepData):
        self.task = task
        self.log = logger.bind(src='Step')
        self._data = data
        self._summary = Counter()
    
    @property
    def data(self):
        return self._data
    
    def __getitem__(self, name: str):
        return getattr(self._data, name)
    
    def __setitem__(self, name: str, value: Any):
        setattr(self._data, name, value)
    
    def get(self, name: str, default: Any = None):
        return getattr(self._data, name, default)
    
    def request(self, user_message: ChatMessage) -> Response:
        client = self.task.client
        self.task.emit('request_started', llm=client.name)
        msg = client(user_message)
        self.task.emit('response_completed', llm=client.name, msg=msg)
        if isinstance(msg.message, ErrorMessage):
            response = Response(message=msg)
            self.log.error('LLM request error', error=msg.content)
        else:
            self._summary.update(msg.usage)
            response = Response.from_message(msg, parse_mcp=self.task.mcp)
        return response

    def process(self, response: Response) -> list[ToolCallResult] | None:
        if isinstance(response.message.message, ErrorMessage):
            return None
        
        if response.task_status:
            self.task.emit('task_status', status=response.task_status)

        if response.code_blocks:
            self.task.blocks.add_blocks(response.code_blocks)
        
        if response.tool_calls:
            toolcall_results = self.task.tool_call_processor.process(self.task, response.tool_calls)
        else:
            toolcall_results = None
        return toolcall_results
    
    def run(self) -> Response:
        max_rounds = self.task.max_rounds
        message_storage = self.task.message_storage
        user_message = self.data.initial_instruction

        while len(self['rounds']) < max_rounds:
            # 请求LLM回复
            response = self.request(user_message)
            self.task.emit('parse_reply_completed', response=response)
            
            # 创建新的Round，包含LLM回复
            round = Round(llm_response=response)
            
            # 处理工具调用
            round.toolcall_results = self.process(response)
            
            # 生成系统反馈消息
            system_feedback = round.get_system_feedback(self.task.prompts)
            if not system_feedback:
                break

            round.system_feedback = message_storage.store(system_feedback)
            self._data.add_round(round)
            user_message = round.system_feedback

        self['end_time'] = time.time()
        return self.data.final_response

    def get_summary(self):
        summary = dict(self._summary)
        summary['elapsed_time'] = int(self['end_time'] - self['start_time'])
        summary['rounds'] = len(self['rounds'])
        summarys = "{rounds} | {elapsed_time}s | Tokens: {input_tokens}/{output_tokens}/{total_tokens}".format(**summary)
        return {'summary': summarys}
    
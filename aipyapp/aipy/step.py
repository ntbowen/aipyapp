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
from .toolcalls import ToolCallResult
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
    system_feedback: UserMessage | None = None

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
    
class StepData(BaseModel):
    # 用户的初始指令作为Step级别的字段
    initial_instruction: ChatMessage
    instruction: str  # 保持向后兼容
    title: str | None = None
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    
    # 每个Round包含完整的对话+执行循环  
    rounds: List[Round] = Field(default_factory=list)
    
    # LLM的最终回复作为Step级别的字段
    final_response: Response | None = None
    
    events: List[BaseEvent.get_subclasses_union()] = Field(default_factory=list)
    
    @property
    def result(self):
        return self.final_response
    
    def add_round(self, round: Round):
        self.rounds.append(round)
        # 更新最终回复
        self.final_response = round.llm_response

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
    
    def run(self, user_message: UserMessage) -> Response:
        max_rounds = self.task.max_rounds
        message_storage = self.task.message_storage
        
        # 使用已经存储的初始指令
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
            if system_feedback:
                round.system_feedback = message_storage.store(system_feedback)
            
            # 添加Round到Step
            self._data.add_round(round)
            
            if not round.should_continue():
                break

            # 下一轮使用系统反馈作为用户输入
            user_message = round.system_feedback

        self['end_time'] = time.time()
        return response

    def get_summary(self):
        summary = dict(self._summary)
        summary['elapsed_time'] = int(time.time() - self['start_time'])
        summary['rounds'] = len(self['rounds'])
        summarys = "{rounds} | {time}s/{elapsed_time}s | Tokens: {input_tokens}/{output_tokens}/{total_tokens}".format(**summary)
        return {'summary': summarys}
    
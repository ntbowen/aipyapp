#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import TYPE_CHECKING
from loguru import logger

from .. import __version__
from ..llm import ModelCapability, AIMessage
from .chat import ChatMessage

if TYPE_CHECKING:
    from .task import Task

class LineReceiver(list):
    def __init__(self):
        super().__init__()
        self.buffer = ""

    @property
    def content(self):
        return '\n'.join(self)
    
    def feed(self, data: str):
        self.buffer += data
        new_lines = []

        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line:
                self.append(line)
                new_lines.append(line)

        return new_lines
    
    def empty(self):
        return not self and not self.buffer
    
    def done(self):
        buffer = self.buffer
        if buffer:
            self.append(buffer)
            self.buffer = ""
        return buffer

class StreamProcessor:
    """流式数据处理器，负责处理 LLM 流式响应并发送事件"""
    
    def __init__(self, task, name):
        self.task = task
        self.name = name
        self.lr = LineReceiver()
        self.lr_reason = LineReceiver()

    @property
    def content(self):
        return self.lr.content
    
    @property
    def reason(self):
        return self.lr_reason.content
    
    def __enter__(self):
        """支持上下文管理器协议"""
        self.task.emit('stream_started', llm=self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持上下文管理器协议"""
        if self.lr.buffer:
            self.process_chunk('\n')        
        self.task.emit('stream_completed', llm=self.name)
    
    def process_chunk(self, content, *, reason=False):
        """处理流式数据块并发送事件"""
        if not content: 
            return

        # 处理思考内容的结束
        if not reason and self.lr.empty() and not self.lr_reason.empty():
            line = self.lr_reason.done()
            if line:
                self.task.emit('stream', llm=self.name, lines=[line, "\n\n----\n\n"], reason=True)

        # 处理当前数据块
        lr = self.lr_reason if reason else self.lr
        lines = lr.feed(content)
        if not lines:
            return
        
        # 过滤掉特殊注释行
        lines2 = [line for line in lines if not (line.startswith('<!-- Block-') or line.startswith('<!-- ToolCall:'))]
        if lines2:
            self.task.emit('stream', llm=self.name, lines=lines2, reason=reason)

class Client:
    def __init__(self, task: 'Task'):
        self.manager = task.client_manager
        self.current = self.manager.current
        self.task = task
        self.extra_headers = {'Aipy-Task-ID': f'{task.task_id}/{__version__}'}

        # 接收外部传入的上下文管理器
        self.context_manager = task.context_manager
        self.storage = task.message_storage
        self.log = logger.bind(src='Client', name=self.current.name)

    @property
    def name(self):
        return self.current.name
    
    def use(self, name):
        client = self.manager.get_client(name)
        if client and client.usable():
            self.current = client
            self.log = logger.bind(src='client', name=self.current.name)
            return True
        return False
    
    def has_capability(self, message: ChatMessage) -> bool:
        # 判断 content 需要什么能力
        if isinstance(message.content, str):
            return True
        
        #TODO: 不应该硬编码字符串
        if self.current.kind == 'trust':
            return True
        
        model = self.current.model
        model = model.rsplit('/', 1)[-1]
        model_info = self.manager.get_model_info(model)
        if not model_info:
            self.log.error(f"Model info not found for {model}")
            return False
                
        capabilities = set()
        for item in message.content:
            if item.type == 'image_url':
                capabilities.add(ModelCapability.IMAGE_INPUT)
            if item.type == 'file':
                capabilities.add(ModelCapability.FILE_INPUT)
            if item.type == 'text':
                capabilities.add(ModelCapability.TEXT)
        
        return any(capability in model_info.capabilities for capability in capabilities)
    
    def __call__(self, user_message: ChatMessage) -> ChatMessage:
        client = self.current
        stream_processor = StreamProcessor(self.task, client.name)
        
        messages = self.context_manager.get_messages()
        messages.append(user_message)
        msg = client([msg.dict() for msg in messages], stream_processor=stream_processor, extra_headers=self.extra_headers)
        msg = self.storage.store(msg)
        if isinstance(msg.message, AIMessage):
            self.context_manager.add_chat(user_message, msg)
        return msg

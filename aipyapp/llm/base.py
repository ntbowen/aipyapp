#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from enum import Enum
from collections import Counter
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from .. import T

class TextItem(BaseModel):
    type: Literal['text'] = 'text'
    text: str

class ImageUrl(BaseModel):
    url: str

class ImageItem(BaseModel):
    type: Literal['image-url'] = 'image-url'
    image_url: ImageUrl

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    ERROR = "error"

class Message(BaseModel):
    role: MessageRole
    content: str

    def dict(self):
        return {'role': self.role.value, 'content': self.content}
    
class UserMessage(Message):
    role: Literal[MessageRole.USER] = MessageRole.USER
    content: Union[str, List[Union[TextItem, ImageItem]]]

    @property
    def content_str(self):
        if isinstance(self.content, str):
            return self.content
        contents = []
        for item in self.content:
            if item.type == 'text':
                contents.append(item.text)
            elif item.type == 'image-url':
                contents.append(item.image_url.url)
        return '\n'.join(contents)
    
class SystemMessage(Message):
    role: Literal[MessageRole.SYSTEM] = MessageRole.SYSTEM

class AIMessage(Message):
    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT
    reason: str | None = None
    usage: Counter = Field(default_factory=Counter)
    
class ErrorMessage(Message):
    role: Literal[MessageRole.ERROR] = MessageRole.ERROR

class BaseClient(ABC):
    MODEL = None
    BASE_URL = None
    TEMPERATURE = 0.5

    def __init__(self, config):
        self.name = config['name']
        self.log = logger.bind(src='llm', name=self.name)
        self.console = None
        self.config = config
        self.kind = config.get("type", "openai")
        self.max_tokens = config.get("max_tokens")
        self._model = config.get("model") or self.MODEL
        self._timeout = config.get("timeout")
        self._api_key = config.get("api_key")
        self._base_url = self.get_base_url()
        self._stream = config.get("stream", True)
        self._tls_verify = bool(config.get("tls_verify", True))
        self._client = None
        params = self.get_params()
        params.update(config.get("params", {}))
        self._params = params
        self._temperature = params.get("temperature") or self.TEMPERATURE

    @property
    def model(self):
        return self._model
    
    @property
    def base_url(self):
        return self._base_url
    
    def __repr__(self):
        return f"{self.name}/{self.kind}:{self._model}"
    
    def get_params(self):
        return {}
    
    def get_base_url(self):
        return self.config.get("base_url") or self.BASE_URL
    
    def usable(self):
        return self._model
    
    def _get_client(self):
        return self._client
    
    @abstractmethod
    def get_completion(self, messages: list[Dict[str, Any]], **kwargs) -> AIMessage:
        pass
        
    def _prepare_messages(self, messages: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        return messages
    
    @abstractmethod
    def _parse_usage(self, response) -> Counter:
        pass

    @abstractmethod
    def _parse_stream_response(self, response, stream_processor) -> AIMessage:
        pass

    @abstractmethod
    def _parse_response(self, response) -> AIMessage:
        pass
    
    def __call__(self, messages: list[Dict[str, Any]], stream_processor=None, **kwargs) -> AIMessage | ErrorMessage:
        messages = self._prepare_messages(messages)
        start = time.time()
        try:
            response = self.get_completion(messages, **kwargs)
        except Exception as e:
            self.log.exception(f"{self.name} API Call failed", e=e)
            return ErrorMessage(content=str(e))

        if self._stream:
            msg = self._parse_stream_response(response, stream_processor)
        else:
            msg = self._parse_response(response)

        msg.usage['time'] = int(time.time() - start)
        if not msg.content:
            self.log.warning("Got empty LLM response")
        return msg
    
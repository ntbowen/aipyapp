import hashlib
import base64
from collections import Counter
from typing import Optional, List, Union, Dict

from pydantic import BaseModel, Field

from ..llm import MessageRole, AIMessage, UserMessage, SystemMessage, ErrorMessage
from .types import InstanceTrackerMixin

class ChatMessage(InstanceTrackerMixin, BaseModel):
    id: str
    message: Union[AIMessage, UserMessage, SystemMessage, ErrorMessage] = Field(exclude=True, default=None)

    def __hash__(self):
        return hash(self.id)
    
    def model_post_init(self, __context):
        if __context and 'message_storage' in __context:
            message_storage = __context['message_storage']
            self.message = message_storage.get(self.id)
            #print(f'Recover message {self.id} from storage: {self.message is not None}')

    @property
    def role(self):
        return self.message.role if self.message is not None else None
    
    @property
    def content(self):
        return self.message.content if self.message is not None else None
    
    @property
    def reason(self):
        return self.message.reason if self.message is not None else None
    
    @property
    def usage(self):
        return self.message.usage if self.message is not None else Counter()
    
    @property
    def total_tokens(self):
        if self.role == MessageRole.ASSISTANT and self.message.usage:
            return self.message.usage.get('total_tokens', 0)
        return None
    
    def dict(self) -> dict:
        return self.message.dict() if self.message is not None else {}
    
class MessageStorage(BaseModel):
    messages: Dict[str, Union[AIMessage, UserMessage, SystemMessage, ErrorMessage]] = Field(default_factory=dict)

    def __len__(self):
        return len(self.messages)

    def __contains__(self, id: str) -> bool:
        return id in self.messages
    
    def _compute_id(self, message: Union[AIMessage, UserMessage, SystemMessage, ErrorMessage]) -> str:
        """Compute short hash of message role + content using Base64-encoded 8-byte SHA-1"""
        content_str = f"{message.role}:{message.content}"
        hash_bytes = hashlib.sha1(content_str.encode('utf-8')).digest()[:8]
        return base64.urlsafe_b64encode(hash_bytes).decode().rstrip('=')
    
    def store(self, message: Union[AIMessage, UserMessage, SystemMessage, ErrorMessage]) -> ChatMessage:
        id = self._compute_id(message)
        try:
            message = self.messages[id]
        except KeyError:
            self.messages[id] = message
        return ChatMessage(id=id, message=message)

    def get(self, id: str) -> Optional[Union[AIMessage, UserMessage, SystemMessage, ErrorMessage]]:
        return self.messages.get(id)

class ChatMessages(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    summary: Counter = Field(default_factory=Counter)

    def __len__(self):
        return len(self.messages)
    
    def __iter__(self):
        return iter(self.messages)
    
    @property
    def total_tokens(self):
        return self.usage.get('total_tokens', 0)
    
    def append(self, message: ChatMessage):
        self.messages.append(message)
        if message.role == MessageRole.ASSISTANT:
            self.summary += message.usage

    def get_summary(self) -> dict:
        ret = {'time': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        summary = self.summary
        if not summary:
            summary = Counter({'time': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})
        ret.update(dict(summary))
        return ret

    def update_summary(self) -> Counter:
        summary = self.summary
        summary.clear()
        for message in self.messages:
            if message.role == MessageRole.ASSISTANT:
                summary += message.usage
        return summary
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Literal, Union
from pydantic import BaseModel, Field, ConfigDict
import time

from loguru import logger

from .response import Response, TaskCompleted, TaskCannotContinue
from ..exec import ExecResult, ProcessResult, PythonResult
from .blocks import CodeBlock
from .chat import ChatMessage
from .toolcalls import ToolCall, ToolCallResult

# ==================== Strongly-Typed Event System ====================

# Base event classes
class BaseEvent(BaseModel):
    """Base event class for all events"""
    name: str = Field(..., title="Event Name", description="Unique identifier for the event type")
    timestamp: float = Field(
        default_factory=lambda: time.time(),
        title="Timestamp", 
        description="Unix timestamp when the event occurred"
    )
    
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True, discriminator='name')

    @classmethod
    def get_subclasses_union(cls):
        return Union[tuple(cls.__subclasses__())]

class TypedEvent:
    """Wrapper for backward compatibility with existing Event class"""
    def __init__(self, typed_event: BaseEvent):
        self._typed_event = typed_event
        self.name = typed_event.name
        # Create data dict for compatibility with existing code
        self.data = {}
        for field_name, field_value in typed_event.model_dump().items():
            if field_name != 'name':  # name is already set separately
                self.data[field_name] = field_value
    
    @property
    def typed_event(self):
        return self._typed_event
    
    def __getattr__(self, name: str):
        # Prefer attributes from typed event
        if hasattr(self._typed_event, name):
            return getattr(self._typed_event, name)
        # Fall back to data dict
        return self.data.get(name)
    
    def __str__(self):
        return f"{self.name}: {self.data}"

class EventFactory:
    """Factory class for creating strongly-typed events"""
    
    _event_registry: Dict[str, type] = {}
    
    @classmethod
    def create_event(cls, event_name: str, **kwargs) -> BaseEvent:
        """Create a strongly-typed event instance"""
        event_class = cls._event_registry.get(event_name)
        if event_class:
            try:
                return event_class(**kwargs)
            except Exception as e:
                # Fall back to base event if typed creation fails
                logger.warning(f"Failed to create typed event {event_name}: {e}")
                return BaseEvent(name=event_name, **kwargs)
        
        # Use base event for unregistered event types
        return BaseEvent(name=event_name, **kwargs)
    
    @classmethod
    def register_event(cls, event_name: str, event_class: type):
        """Register a new event type"""
        cls._event_registry[event_name] = event_class
    
    @classmethod
    def get_registered_events(cls) -> Dict[str, type]:
        """Get all registered event types"""
        return cls._event_registry.copy()
    
    @classmethod
    def is_registered(cls, event_name: str) -> bool:
        """Check if an event type is registered"""
        return event_name in cls._event_registry
    
    @classmethod
    def get_event_class(cls, event_name: str) -> type:
        """Get the event class for a given event name"""
        return cls._event_registry.get(event_name, BaseEvent)
    
# Updated EventBus with strongly-typed event support
class TypedEventBus:
    """Event bus with strongly-typed event support"""
    def __init__(self):
        self._listeners: Dict[str, List] = {}
        self._eb_logger = logger.bind(src=self.__class__.__name__)
    
    def emit_event(self, event: BaseEvent):
        """Emit a strongly-typed event"""
        typed_event = TypedEvent(event)
        for handler in self._listeners.get(event.name, []):
            try:
                handler(typed_event)
            except Exception as e:
                self._eb_logger.exception(f"Error handling event {event.name}")

    def emit(self, event_name: str, **kwargs) -> BaseEvent:
        """Emit a strongly-typed event"""
        # Create typed event object
        event = EventFactory.create_event(event_name, **kwargs)
        self.emit_event(event)
        return event
    
    def on_event(self, event_name: str, handler):
        """Register an event handler"""
        self._listeners.setdefault(event_name, []).append(handler)
    
    def add_listener(self, obj):
        """Add an event listener object"""
        count = 0
        if hasattr(obj, 'get_handlers'):
            handlers = obj.get_handlers()
            for event_name, handler in handlers.items():
                self.on_event(event_name, handler)
                count += 1
        else:
            # Auto-discover on_* methods
            for attr_name in dir(obj):
                if attr_name.startswith('on_') and callable(getattr(obj, attr_name)):
                    event_name = attr_name[3:]  # Remove 'on_' prefix
                    handler = getattr(obj, attr_name)
                    self.on_event(event_name, handler)
                    count += 1
        
        if count > 0:
            self._eb_logger.info(f"Registered {count} events for {obj.__class__.__name__}")

# ==================== Event Type Definitions ====================

# ==================== Task Lifecycle Events ====================

class TaskStartedEvent(BaseEvent):
    """Event fired when a task starts"""
    name: Literal["task_started"] = "task_started"
    instruction: str = Field(..., title="Instruction", description="User instruction for the task")
    title: Optional[str] = Field(None, title="Title", description="Optional title for the task")
    task_id: str = Field(..., title="Task ID", description="Unique identifier for the task")
    parent_id: str|None = Field(..., title="Parent ID", description="Unique identifier for the parent task")
    
class TaskCompletedEvent(BaseEvent):
    """Event fired when a task ends"""
    name: Literal["task_completed"] = "task_completed"
    path: Optional[str] = Field(None, title="Path", description="Path where task results are saved")
    task_id: str = Field(..., title="Task ID", description="Unique identifier for the task")
    parent_id: str|None = Field(..., title="Parent ID", description="Unique identifier for the parent task")
 
class StepStartedEvent(BaseEvent):
    """Event fired when a conversation round starts"""
    name: Literal["step_started"] = "step_started"
    instruction: str = Field(..., title="Instruction", description="User instruction for this round")
    step: int = Field(..., title="Step", description="Step number")
    title: Optional[str] = Field(None, title="Title", description="Optional title for this step")

class StepCompletedEvent(BaseEvent):
    """Event fired when a conversation round ends"""
    name: Literal["step_completed"] = "step_completed"
    summary: Dict[str, Any] = Field(..., title="Summary", description="Step execution summary")
    response: Optional[Response] = Field(None, title="Response", description="Final response content")

class TaskStatusEvent(BaseEvent):
    """Event fired when task status changes"""
    name: Literal["task_status"] = "task_status"
    status: TaskCompleted | TaskCannotContinue = Field(..., title="Status", description="Current task status information")

# ==================== LLM Interaction Events ====================

class RequestStartedEvent(BaseEvent):
    """Event fired when LLM query starts"""
    name: Literal["request_started"] = "request_started"
    llm: str = Field(..., title="LLM", description="Name of the LLM being queried")

class ResponseCompletedEvent(BaseEvent):
    """Event fired when LLM response is complete"""
    name: Literal["response_completed"] = "response_completed"
    llm: str = Field(..., title="LLM", description="Name of the LLM that responded")
    msg: ChatMessage = Field(..., title="Message", description="LLM Response message object")

class StreamStartedEvent(BaseEvent):
    """Event fired when LLM streaming starts"""
    name: Literal["stream_started"] = "stream_started"
    llm: Optional[str] = Field(None, title="LLM", description="Name of the streaming LLM")

class StreamCompletedEvent(BaseEvent):
    """Event fired when LLM streaming ends"""
    name: Literal["stream_completed"] = "stream_completed"
    llm: Optional[str] = Field(None, title="LLM", description="Name of the streaming LLM")

class StreamEvent(BaseEvent):
    """Event fired for each streaming chunk"""
    name: Literal["stream"] = "stream"
    llm: Optional[str] = Field(None, title="LLM", description="Name of the streaming LLM")
    lines: Optional[List[str]] = Field(None, title="Lines", description="Streaming content lines")
    reason: Optional[bool] = Field(None, title="Reason", description="Whether this chunk contains reasoning")

# ==================== Message Parsing Events ====================

class ParseReplyCompletedEvent(BaseEvent):
    """Event fired when parsing LLM response"""
    name: Literal["parse_reply_completed"] = "parse_reply_completed"
    response: Response = Field(..., title="Response", description="Parsed response object")

# ==================== Code Execution Events ====================

class ExecStartedEvent(BaseEvent):
    """Event fired when code execution starts"""
    name: Literal["exec_started"] = "exec_started"
    block: CodeBlock = Field(..., title="Block", description="Code block being executed")

class ExecCompletedEvent(BaseEvent):
    """Event fired when code execution completes"""
    name: Literal["exec_completed"] = "exec_completed"
    result: Union[ExecResult, ProcessResult, PythonResult] = Field(..., title="Result", description="Execution result data")
    block: CodeBlock = Field(..., title="Block", description="Code block that was executed")

# ==================== Code Editing Events ====================

class EditStartedEvent(BaseEvent):
    """Event fired when code editing starts"""
    name: Literal["edit_start"] = "edit_start"
    block_name: str = Field(..., title="Block Name", description="Name of the code block being edited")
    old: str = Field(..., title="Old", description="Old code string")
    new: str = Field(..., title="New", description="New code string")
    replace_all: bool = Field(..., title="Replace All", description="Whether to replace all occurrences")

class EditCompletedEvent(BaseEvent):
    """Event fired when code editing completes"""
    name: Literal["edit_completed"] = "edit_completed"
    block_name: str = Field(..., title="Block Name", description="Name of the code block being edited")
    success: bool = Field(..., title="Success", description="Whether the edit operation succeeded")
    new_version: Optional[int] = Field(None, title="New Version", description="New version number of the code block")

# ==================== Tool Calling Events ====================

class ToolCallStartedEvent(BaseEvent):
    """Event fired when tool call starts"""
    name: Literal["tool_call_started"] = "tool_call_started"
    tool_call: ToolCall = Field(..., title="Tool Call", description="Tool call object")

class ToolCallCompletedEvent(BaseEvent):
    """Event fired when tool call completes"""
    name: Literal["tool_call_completed"] = "tool_call_completed"
    result: ToolCallResult = Field(..., title="Result", description="Tool call result object")

# ==================== Function Calling Events ====================

class FunctionCallStartedEvent(BaseEvent):
    """Event fired when a function is called"""
    name: Literal["function_call_started"] = "function_call_started"
    funcname: str = Field(..., title="Function Name", description="Name of the function being called")
    kwargs: Optional[Dict[str, Any]] = Field(None, title="Arguments", description="Function call arguments")

class FunctionCallCompletedEvent(BaseEvent):
    """Event fired when function call completes"""
    name: Literal["function_call_completed"] = "function_call_completed"
    funcname: str = Field(..., title="Function Name", description="Name of the function that was called")
    kwargs: Optional[Dict[str, Any]] = Field(None, title="Arguments", description="Function call arguments")
    result: Optional[Any] = Field(None, title="Result", description="Function return value")
    success: bool = Field(..., title="Success", description="Whether the function call succeeded")
    error: Optional[str] = Field(None, title="Error", description="Error message if call failed")
    exception: Optional[Exception] = Field(None, title="Exception", description="Exception object if call failed")

# ==================== Runtime Events ====================

class RuntimeMessageEvent(BaseEvent):
    """Event fired when runtime sends a message"""
    name: Literal["runtime_message"] = "runtime_message"
    message: str = Field(..., title="Message", description="Runtime message content")
    status: Optional[str] = Field(None, title="Status", description="Message status level (info, warning, error)")

class RuntimeInputEvent(BaseEvent):
    """Event fired when runtime requests input"""
    name: Literal["runtime_input"] = "runtime_input"
    prompt: str = Field(..., title="Prompt", description="Input prompt message")

class ShowImageEvent(BaseEvent):
    """Event fired when an image should be displayed"""
    name: Literal["show_image"] = "show_image"
    path: Optional[str] = Field(None, title="Path", description="Local file path to the image")
    url: Optional[str] = Field(None, title="URL", description="URL to the image")

# ==================== System Events ====================

class ExceptionEvent(BaseEvent):
    """Event fired when an exception occurs"""
    name: Literal["exception"] = "exception"
    msg: str = Field(..., title="Message", description="Error context message")
    exception: Exception = Field(..., title="Exception", description="The exception object")

class UploadResultEvent(BaseEvent):
    """Event fired when upload operation completes"""
    name: Literal["upload_result"] = "upload_result"
    status_code: int = Field(..., title="Status Code", description="HTTP status code of the upload")
    url: Optional[str] = Field(None, title="URL", description="URL of the uploaded content")

class StepCleanupCompletedEvent(BaseEvent):
    """Event fired when step cleanup completes"""
    name: Literal["step_cleanup_completed"] = "step_cleanup_completed"
    cleaned_messages: int = Field(..., title="Cleaned Messages", description="Number of messages cleaned up")
    remaining_messages: int = Field(..., title="Remaining Messages", description="Number of messages remaining after cleanup")
    tokens_saved: int = Field(..., title="Tokens Saved", description="Number of tokens saved by cleanup")
    tokens_remaining: int = Field(..., title="Tokens Remaining", description="Number of tokens remaining after cleanup")

# ==================== Event Factory ====================

# Register all event types with the factory
def _register_all_events():
    """Register all event types with the EventFactory"""
    subclasses = BaseEvent.__subclasses__()
    for event_class in subclasses:
        event_name = event_class.model_fields['name'].default
        EventFactory.register_event(event_name, event_class)

# Register all events when module is imported
_register_all_events()
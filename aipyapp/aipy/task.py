#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import uuid
import zlib
import base64
from typing import Any, Dict, List, Union, TYPE_CHECKING
from pathlib import Path
from importlib.resources import read_text

import requests
from pydantic import BaseModel, Field, ValidationError, field_serializer, field_validator
from loguru import logger

from .. import T, __respkg__, Stoppable, TaskPlugin
from ..exec import BlockExecutor
from ..llm import SystemMessage, UserMessage
from .runtime import CliPythonRuntime
from .utils import safe_rename, validate_file
from .events import TypedEventBus, BaseEvent
from .multimodal import MMContent   
from .context import ContextManager, ContextData
from .toolcalls import ToolCallProcessor
from .chat import MessageStorage, ChatMessage
from .step import Step, StepData
from .blocks import CodeBlocks
from .client import Client
from .response import Response

if TYPE_CHECKING:
    from .taskmgr import TaskManager

MAX_ROUNDS = 16
TASK_VERSION = 20250818

CONSOLE_WHITE_HTML = read_text(__respkg__, "console_white.html")
CONSOLE_CODE_HTML = read_text(__respkg__, "console_code.html")

class TaskError(Exception):
    """Task 异常"""
    pass

class TaskInputError(TaskError):
    """Task 输入异常"""
    def __init__(self, message: str, original_error: Exception = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)

class TastStateError(TaskError):
    """Task 状态异常"""
    def __init__(self, message: str, **kwargs):
        self.message = message
        self.data = kwargs
        super().__init__(self.message)

class TaskData(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    version: int = Field(default=TASK_VERSION, frozen=True)
    steps: List[StepData] = Field(default_factory=list)
    blocks: CodeBlocks = Field(default_factory=CodeBlocks)
    context: ContextData = Field(default_factory=ContextData)
    message_storage: MessageStorage = Field(default_factory=MessageStorage)
    events: List[BaseEvent.get_subclasses_union()] = Field(default_factory=list)
    session: Dict[str, Any] = Field(default_factory=dict)

    @field_serializer('events')
    def serialize_events(self, events: List, _info):
        """序列化时压缩 events 字段"""
        if not events:
            return None
        # 将 events 序列化为 JSON 字符串
        json_str = json.dumps([e.model_dump() for e in events], ensure_ascii=False)
        # 使用 zlib 压缩
        compressed = zlib.compress(json_str.encode('utf-8'), level=9)
        # Base64 编码以便存储为字符串
        return base64.b64encode(compressed).decode('ascii')

    @field_validator('events', mode='before')
    @classmethod
    def deserialize_events(cls, v):
        """反序列化时解压 events 字段"""
        if v is None or v == []:
            return []
        if isinstance(v, list):
            # 已经是列表，直接返回（可能来自内存中的对象）
            return v
        if isinstance(v, str):
            # 从压缩字符串恢复
            try:
                compressed = base64.b64decode(v.encode('ascii'))
                json_str = zlib.decompress(compressed).decode('utf-8')
                events_data = json.loads(json_str)
                # 需要重新构造事件对象
                return events_data  # Pydantic 会自动验证和构造
            except Exception as e:
                logger.warning(f"Failed to decompress events: {e}")
                return []
        return v

    def add_step(self, step: StepData):
        self.steps.append(step)

class Task(Stoppable):
    def __init__(self, manager: TaskManager, data: TaskData | None = None, parent: Task | None = None):
        super().__init__()
        data = data or TaskData()

        # Phase 1: Initialize basic attributes (no dependencies)
        self.data = data
        self.parent = parent
        self.task_id = data.id
        self.manager = manager
        self.settings = manager.settings
        self.log = logger.bind(src='task', id=self.task_id)
        self.cwd = manager.cwd / self.task_id if not parent else parent.cwd / self.task_id
        self.gui = manager.settings.gui
        self._saved = False
        self.max_rounds = manager.settings.get('max_rounds', MAX_ROUNDS)
        self.role = manager.role_manager.current_role
        
        # Phase 2: Initialize data objects (minimal dependencies)
        if parent:
            pass
            #data.context = parent.context.model_copy(deep=True)
            #data.message_storage = parent.message_storage.model_copy(deep=True)
            #data.blocks = parent.blocks.model_copy(deep=True)

        self.blocks = data.blocks
        self.message_storage = data.message_storage
        self.context = data.context
        self.events = data.events

        # session: 子任务共享父任务的 session 引用，根任务使用 TaskData 中的 session
        if parent:
            self.session = parent.session
        else:
            self.session = data.session
        
        # Phase 3: Initialize managers and processors (depend on Phase 2)
        self.event_bus = TypedEventBus() if not parent else parent.event_bus
        self.context_manager = ContextManager(
            self.message_storage,
            self.context,
            manager.settings.get('context_manager')
        )
        self.tool_call_processor = ToolCallProcessor() if not parent else parent.tool_call_processor
        
        # Phase 4: Initialize display (depends on event_bus)
        if not parent:
            if manager.display_manager:
                self.display = manager.display_manager.create_display_plugin()
                self.event_bus.add_listener(self.display)
            else:
                self.display = None
        else:
            self.display = parent.display
        
        # Phase 5: Initialize execution components (depend on task)
        self.mcp = manager.mcp
        self.prompts = manager.prompts
        self.client_manager = manager.client_manager
        self.runtime = CliPythonRuntime(self)
        self.runner = BlockExecutor()
        self.runner.set_python_runtime(self.runtime)
        self.client = Client(self)
        
        # Phase 6: Initialize cleaners (depend on context_manager)
        self.step_cleaner = SimpleStepCleaner(self.context_manager)
        
        # Phase 7: Initialize plugins (depend on runtime and event_bus)
        if not parent:
            self._initialize_plugins(manager)
        else:
            self.plugins = parent.plugins
        
        # Phase 8: Initialize steps last (depend on almost everything)
        self.steps: List[Step] = [Step(self, step_data) for step_data in data.steps]
    
    def _initialize_plugins(self, manager: TaskManager):
        """Separate method to initialize plugins, improving clarity and testability"""
        plugins: dict[str, TaskPlugin] = {}
        for plugin_name, plugin_data in self.role.plugins.items():
            plugin = manager.plugin_manager.create_task_plugin(plugin_name, plugin_data)
            if not plugin:
                self.log.warning(f"Create task plugin {plugin_name} failed")
                continue
            self.runtime.register_plugin(plugin)
            self.event_bus.add_listener(plugin)
            plugins[plugin_name] = plugin
        self.plugins = plugins

    @property
    def instruction(self):
        return self.steps[0].data.instruction if self.steps else None

    def use(self, llm: str) -> bool:
        """ for cmd_llm use
        """
        return self.client.use(llm)

    def run_block(self, name: str) -> bool:
        """ for cmd_block run
        """
        block = self.blocks.get(name)
        if not block:
            return False
        result = self.runner(block)
        self.emit('exec_completed', result=result, block=block)
        return True

    def emit(self, event_name: str, **kwargs):
        event = self.event_bus.emit(event_name, **kwargs)
        self.events.append(event)
        return event

    def get_system_message(self) -> ChatMessage:
        params = {}
        if self.mcp:
            params['mcp_tools'] = self.mcp.get_tools_prompt()
        params['util_functions'] = self.runtime.get_builtin_functions()
        params['tool_functions'] = self.runtime.get_plugin_functions()
        params['role'] = self.role
        system_prompt = self.prompts.get_default_prompt(**params)
        msg = SystemMessage(content=system_prompt)
        return self.message_storage.store(msg)
    
    def new_step(self, step_data: StepData) -> Step:
        """ 准备一个新的Step
        """
        self.data.add_step(step_data)
        step = Step(self, step_data)
        self.steps.append(step)
        return step
    
    def delete_step(self, index: int) -> bool:
        """删除指定索引的Step并清理其上下文消息"""
        if index < 0 or index >= len(self.steps):
            self.log.warning(f"Invalid step index: {index}")
            return False
            
        if index == 0:
            self.log.warning("Cannot delete Step 0")
            return False
            
        # 获取要删除的Step
        step_to_delete = self.steps[index]
        step_info = step_to_delete.data.instruction[:50] + "..." if len(step_to_delete.data.instruction) > 50 else step_to_delete.data.instruction
        
        try:
            # 先清理上下文中的相关消息
            cleaned_count, remaining_messages, tokens_saved, tokens_remaining = self.step_cleaner.delete_step(step_to_delete)  # noqa: E501
            
            # 然后从步骤列表中删除
            self.steps.pop(index)
            
            self.log.info(f"Deleted step {index}: {step_info}")
            self.log.info(f"Context cleanup: {cleaned_count} messages deleted, {tokens_saved} tokens saved")
            self.emit('step_deleted', 
                     step_index=index, 
                     step_info=step_info,
                     cleaned_messages=cleaned_count,
                     tokens_saved=tokens_saved)
            
            return True
            
        except Exception as e:
            self.log.error(f"Failed to delete step {index}: {e}")
            return False

    def get_status(self):
        return {
            'llm': self.client.name,
            'blocks': len(self.blocks),
            'steps': len(self.steps),
        }

    @classmethod
    def from_file(cls, path: Union[str, Path], manager: TaskManager) -> 'Task':
        """从文件创建 TaskState 对象"""
        path = Path(path)
        validate_file(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
                try:
                    model_context = {'message_storage': MessageStorage.model_validate(data['message_storage'])}
                except Exception:
                    model_context = None

                task_data = TaskData.model_validate(data, context=model_context)
                task = cls(manager, task_data)
                logger.info('Loaded task state from file', path=str(path), task_id=task.task_id)
                return task
        except json.JSONDecodeError as e:
            raise TaskError(f'Invalid JSON file: {e}') from e
        except ValidationError as e:
            raise TaskError(f'Invalid task state: {e.errors()}') from e
        except Exception as e:
            raise TaskError(f'Failed to load task state: {e}') from e
    
    def to_file(self, path: Union[str, Path]) -> None:
        """保存任务状态到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                data = self.data
                f.write(data.model_dump_json(indent=2, exclude_none=True))
            self.log.info('Saved task state to file', path=str(path))
        except Exception as e:
            self.log.exception('Failed to save task state', path=str(path))
            raise TaskError(f'Failed to save task state: {e}') from e
        
    def _auto_save(self):
        """自动保存任务状态"""
        # 如果任务目录不存在，则不保存
        cwd = self.cwd
        if not cwd.exists():
            self.log.warning('Task directory not found, skipping save')
            return
        
        try:
            self.to_file(cwd / "task.json")
            
            display = self.display
            if display:
                filename = cwd / "console.html"
                display.save(filename, clear=False, code_format=CONSOLE_WHITE_HTML)
            
            self._saved = True
            self.log.info('Task auto saved')
        except Exception as e:
            self.log.exception('Error saving task')
            self.emit('exception', msg='save_task', exception=e)

    def done(self):
        if not self.steps or not self.cwd.exists():
            self.log.warning('Task not started, skipping save')
            return
        
        if not self._saved:
            self.log.warning('Task not saved, trying to save')
            self._auto_save()

        try:
            newname = safe_rename(self.cwd, self.instruction)
        except Exception:
            self.log.exception('Failed to rename task directory', path=str(self.cwd))
            newname = self.cwd

        self.log.info('Task done', path=newname)
        self.emit('task_completed', path=str(newname), task_id=self.task_id, parent_id=self.parent.task_id if self.parent else None)
        #self.context.diagnose.report_code_error(self.runner.history)
        if self.settings.get('share_result'):
            self.sync_to_cloud()

    def prepare_user_prompt(self, instruction: str, first_run: bool=False) -> ChatMessage:
        """处理多模态内容并验证模型能力"""
        mmc = MMContent(instruction, base_path=self.cwd.parent)
        try:
            message = mmc.message
        except Exception as e:
            raise TaskInputError(T("Invalid input"), e) from e

        content = message.content
        if isinstance(content, str):
            if first_run:
                content = self.prompts.get_task_prompt(content, gui=self.gui, parent=self.parent)
            else:
                content = self.prompts.get_chat_prompt(content, self.instruction)
            message.content = content
        elif not self.client.has_capability(message):
            raise TaskInputError(T("Current model does not support this content"))

        return self.message_storage.store(message)

    def _auto_compact(self):
       # Step级别的上下文清理
        auto_compact_enabled = self.settings.get('auto_compact_enabled', True)
        if not auto_compact_enabled:
            return
        
        self.log.info("Starting step compact...")
        result = self.step_cleaner.compact_step(self.steps[-1])
        self.log.info(f"Step compact result: {result}")
        cleaned_count, remaining_messages, tokens_saved, tokens_remaining = result
        self.log.info(f"Step compact completed, cleaned_count: {cleaned_count}")
        
        self.emit('step_cleanup_completed', 
                    cleaned_messages=cleaned_count,
                    remaining_messages=remaining_messages,
                    tokens_saved=tokens_saved,
                    tokens_remaining=tokens_remaining)
        self.log.info(f"Step compact completed: {cleaned_count} messages cleaned")

    def run(self, instruction: str, title: str | None = None) -> Response:
        """
        执行自动处理循环，直到 LLM 不再返回代码消息
        instruction: 用户输入的字符串（可包含@file等多模态标记）
        """
        first_run = not self.steps
        user_message = self.prepare_user_prompt(instruction, first_run)
        if first_run:
            self.context_manager.add_message(self.get_system_message())
            self.emit('task_started', instruction=instruction, title=title, task_id=self.task_id, parent_id=self.parent.task_id if self.parent else None)
        else:
            self._auto_compact()

        # We MUST create the task directory here because it could be a resumed task.
        self.cwd.mkdir(exist_ok=True, parents=True)
        os.chdir(self.cwd)
        self._saved = False

        step_data =StepData(
            initial_instruction=user_message,
            instruction=instruction, 
            title=title
        )
        step = self.new_step(step_data)
        self.emit('step_started', instruction=instruction, step=len(self.steps) + 1, title=title)
        response = step.run()
        self.emit('step_completed', summary=step.get_summary(), response=response)

        self._auto_save()
        self.log.info('Step done', rounds=len(step.data.rounds))
        return response

    def run_subtask(self, instruction: str, title: str | None = None, cli=False) -> Response:
        """运行子任务"""
        subtask = Task(self.manager, parent=self)
        response = subtask.run(instruction, title)
        subtask.done()
        if cli:
            self.context_manager.add_chat(UserMessage(content=instruction), response.message)
        return response

    def sync_to_cloud(self):
        """ Sync result
        """
        url = T("https://store.aipy.app/api/work")

        trustoken_apikey = self.settings.get('llm', {}).get('Trustoken', {}).get('api_key')
        if not trustoken_apikey:
            trustoken_apikey = self.settings.get('llm', {}).get('trustoken', {}).get('api_key')
        if not trustoken_apikey:
            return False
        self.log.info('Uploading result to cloud')
        try:
            # Serialize twice to remove the non-compliant JSON type.
            # First, use the json.dumps() `default` to convert the non-compliant JSON type to str.
            # However, NaN/Infinity will remain.
            # Second, use the json.loads() 'parse_constant' to convert NaN/Infinity to str.
            data = json.loads(
                json.dumps({
                    'apikey': trustoken_apikey,
                    'author': os.getlogin(),
                    'instruction': self.instruction,
                    'llm': self.client.name,
                    'runner': self.runner.history,
                }, ensure_ascii=False, default=str),
                parse_constant=str)
            response = requests.post(url, json=data, verify=True,  timeout=30)
        except Exception as e:
            self.emit('exception', msg='sync_to_cloud', exception=e)
            return False

        url = None
        status_code = response.status_code
        if status_code in (200, 201):
            data = response.json()
            url = data.get('url', '')

        self.emit('upload_result', status_code=status_code, url=url)
        return True


class SimpleStepCleaner:
    """Step级别的简化清理器"""
    
    def __init__(self, context_manager):
        self.context_manager = context_manager
        self.log = logger.bind(src='SimpleStepCleaner')
        
    def cleanup_step(self, step) -> tuple[int, int, int, int]:
        """Step完成后的最大化清理：从上下文删除所有Round消息，但保留执行记录
        
        与compact_step的区别：
        - cleanup_step: 删除所有Round消息（最大化清理）
        - compact_step: 只删除失败Round消息（智能清理）
        
        Returns:
            (cleaned_count, remaining_messages, tokens_saved, tokens_remaining)
        """
        if len(step.data.rounds) < 2:
            self.log.info("No enough rounds found in step, skipping cleanup")
            stats = self.context_manager.get_stats()
            return 0, stats['message_count'], 0, stats['total_tokens']
            
        rounds = step.data.rounds
        self.log.info(f"Step has {len(rounds)} rounds, implementing maximum cleanup")
        
        # 获取清理前的统计信息
        stats_before = self.context_manager.get_stats()
        messages_before = stats_before['message_count']
        tokens_before = stats_before['total_tokens']
        
        # 收集除最后一个Round外的所有Round消息ID用于删除
        messages_to_clean = []
        
        for i, round in enumerate(rounds[:-1]):
            # 收集这个Round的所有消息ID
            round.context_deleted = True
            if round.llm_response and round.llm_response.message:
                messages_to_clean.append(round.llm_response.message.id)
            if round.system_feedback:
                messages_to_clean.append(round.system_feedback.id)
                
            self.log.info(f"Will clean Round {i}: {self._get_round_summary(round)}")
        
        self.log.info(f"Will clean {len(messages_to_clean)} messages from {len(rounds)-1} rounds (preserving last round)")
        
        # 执行清理
        if not messages_to_clean:
            self.log.info("No messages need to be cleaned")
            return 0, messages_before, 0, tokens_before
        
        # 执行清理（只清理上下文消息，不影响rounds记录）
        self.context_manager.delete_messages_by_ids(messages_to_clean)
        
        # 获取清理后的统计信息
        stats_after = self.context_manager.get_stats()
        messages_after = stats_after['message_count']
        tokens_after = stats_after['total_tokens']
        
        # 计算清理结果
        cleaned_count = messages_before - messages_after
        tokens_saved = tokens_before - tokens_after
        
        self.log.info(f"Maximum cleanup completed: {cleaned_count} messages cleaned")
        self.log.info(f"Execution records preserved: {len(rounds)} rounds kept")
        self.log.info("Context preserved: initial_instruction + last round")
        self.log.info(f"Messages: {messages_before} -> {messages_after}")
        self.log.info(f"Tokens: {tokens_before} -> {tokens_after} (saved: {tokens_saved})")
        
        return cleaned_count, messages_after, tokens_saved, tokens_after
    
    def compact_step(self, step) -> tuple[int, int, int, int]:
        """智能压缩Step：只清理上下文消息，保留执行记录
        
        基于Round.can_safely_delete()方法智能判断哪些上下文消息可以删除：
        - 删除可安全删除Round对应的上下文消息
        - 保留重要Round对应的上下文消息  
        - 完全保留step.data.rounds（执行历史记录）
        - Step级别的initial_instruction自动保护
        
        Returns:
            (cleaned_count, remaining_messages, tokens_saved, tokens_remaining)
        """
        if len(step.data.rounds) < 2:
            self.log.info("No enough rounds found in step, skipping compact")
            stats = self.context_manager.get_stats()
            return 0, stats['message_count'], 0, stats['total_tokens']
        
        rounds = step.data.rounds
        self.log.info(f"Step has {len(rounds)} rounds, implementing smart compact")
        
        # 获取清理前的统计信息
        stats_before = self.context_manager.get_stats()
        messages_before = stats_before['message_count']
        tokens_before = stats_before['total_tokens']
        
        # 收集需要删除的消息ID
        messages_to_clean = []
        
        # 分析每个Round，收集可删除Round的消息ID
        for i, round in enumerate(rounds):
            if round.can_safely_delete():
                # 收集这个Round的消息ID用于删除
                round.context_deleted = True
                if round.llm_response and round.llm_response.message:
                    messages_to_clean.append(round.llm_response.message.id)
                if round.system_feedback:
                    messages_to_clean.append(round.system_feedback.id)
                    
                self.log.info(f"Will clean Round {i}: {self._get_round_summary(round)}")
            else:
                self.log.info(f"Preserving Round {i}: {self._get_round_summary(round)}")
        
        self.log.info(f"Will clean {len(messages_to_clean)} messages from deletable rounds")
        
        # 执行清理
        if not messages_to_clean:
            self.log.info("No messages need to be cleaned")
            return 0, messages_before, 0, tokens_before
        
        # 执行清理（只清理上下文消息，不影响rounds）
        self.context_manager.delete_messages_by_ids(messages_to_clean)
        
        # 获取清理后的统计信息
        stats_after = self.context_manager.get_stats()
        messages_after = stats_after['message_count']
        tokens_after = stats_after['total_tokens']
        
        # 计算清理结果
        cleaned_count = messages_before - messages_after
        tokens_saved = tokens_before - tokens_after
        
        self.log.info(f"Compact completed: {cleaned_count} messages cleaned")
        self.log.info(f"Execution records preserved: {len(rounds)} rounds kept")
        self.log.info(f"Messages: {messages_before} -> {messages_after}")
        self.log.info(f"Tokens: {tokens_before} -> {tokens_after} (saved: {tokens_saved})")
        
        return cleaned_count, messages_after, tokens_saved, tokens_after
    
    def delete_step(self, step) -> tuple[int, int, int, int]:
        """删除Step时清理所有相关消息：initial_instruction + 所有rounds
        
        Returns:
            (cleaned_count, remaining_messages, tokens_saved, tokens_remaining)
        """
        self.log.info(f"Deleting step context: {step.data.instruction[:50]}...")
        
        # 获取清理前的统计信息
        stats_before = self.context_manager.get_stats()
        messages_before = stats_before['message_count']
        tokens_before = stats_before['total_tokens']
        
        # 收集所有相关消息ID用于删除
        messages_to_clean = []
        
        # 1. 删除initial_instruction
        if step.data.initial_instruction:
            messages_to_clean.append(step.data.initial_instruction.id)
            self.log.info(f"Will delete initial_instruction: {step.data.initial_instruction.id}")
        
        # 2. 删除所有rounds的消息
        for i, round in enumerate(step.data.rounds):
            self.log.info(f"Processing Round {i}: {self._get_round_summary(round)}")
            msg_id = round.llm_response.message.id
            messages_to_clean.append(msg_id)
            self.log.info(f"✅ Will delete Round {i} LLM response: {msg_id}")
                
            # 检查系统反馈
            if round.system_feedback:
                feedback_id = round.system_feedback.id
                messages_to_clean.append(feedback_id)
                self.log.info(f"✅ Will delete Round {i} system feedback: {feedback_id}")
            
            # 标记为删除
            round.context_deleted = True
        
        self.log.info(f"Will delete {len(messages_to_clean)} messages from step deletion")
        
        # 执行清理
        if not messages_to_clean:
            self.log.info("No messages need to be cleaned")
            return 0, messages_before, 0, tokens_before
        
        # 执行清理
        self.log.info(f"Executing delete_messages_by_ids with {len(messages_to_clean)} message IDs")
        deleted_result = self.context_manager.delete_messages_by_ids(messages_to_clean)
        self.log.info(f"delete_messages_by_ids returned: {deleted_result}")
        
        # 获取清理后的统计信息
        stats_after = self.context_manager.get_stats()
        messages_after = stats_after['message_count']
        tokens_after = stats_after['total_tokens']
        
        # 计算清理结果
        cleaned_count = messages_before - messages_after
        tokens_saved = tokens_before - tokens_after
        
        self.log.info(f"Step deletion cleanup completed: {cleaned_count} messages deleted")
        self.log.info(f"Messages: {messages_before} -> {messages_after}")
        self.log.info(f"Tokens: {tokens_before} -> {tokens_after} (saved: {tokens_saved})")
        
        return cleaned_count, messages_after, tokens_saved, tokens_after
    
    def _get_round_summary(self, round) -> str:
        """获取Round的简要描述用于日志"""
        if round.llm_response.errors:
            return "LLM_ERROR"
        elif not round.toolcall_results:
            return "TEXT_ONLY"
        elif all(round._tool_call_failed(tcr) for tcr in round.toolcall_results):
            return f"TOOL_FAILED: {len(round.toolcall_results)} tools"
        else:
            success_count = sum(1 for tcr in round.toolcall_results if not round._tool_call_failed(tcr))
            return f"SUCCESS: {success_count}/{len(round.toolcall_results)} tools"

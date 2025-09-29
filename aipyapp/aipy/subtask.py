#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SubTask系统 - 基于HTML注释ToolCall的子任务实现
"""

from __future__ import annotations
import time
from typing import Dict, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .task import Task

class SubTaskManager:
    """SubTask管理器"""

    def __init__(self, parent_task: 'Task'):
        self.parent_task = parent_task
        self.subtasks: Dict[str, 'Task'] = {}
        self.log = logger.bind(src='SubTaskManager', parent=parent_task.task_id)

    def create_subtask(self, instruction: str, title: Optional[str] = None,
                      inherit_context: bool = False) -> 'Task':
        """创建子任务"""
        # 创建子任务实例
        subtask = self.parent_task.manager.new_task()

        # 设置子任务属性
        subtask.is_subtask = True
        subtask.parent_task_id = self.parent_task.task_id
        subtask.workdir = self.parent_task.cwd
        subtask.cwd = subtask.workdir / subtask.task_id

        # 如果需要继承上下文，复制父任务的关键上下文
        if inherit_context:
            self._inherit_context(subtask)

        # 注册子任务
        self.subtasks[subtask.task_id] = subtask

        self.log.info(f"Created subtask {subtask.task_id}",
                     parent_id=self.parent_task.task_id,
                     title=title,
                     inherit_context=inherit_context)

        return subtask

    def run_subtask(self, subtask: 'Task', instruction: str,
                   title: Optional[str] = None) -> Dict[str, any]:
        """运行子任务并返回结果"""
        self.log.info(f"Starting subtask {subtask.task_id}",
                     parent_id=self.parent_task.task_id)

        # 发送SubTask开始事件
        self.parent_task.emit('subtask_started',
                             subtask_id=subtask.task_id,
                             parent_task_id=self.parent_task.task_id,
                             instruction=instruction,
                             title=title,
                             inherit_context=False)

        start_time = time.time()

        # 在运行前，设置一个钩子来捕获最后一个LLM响应
        captured_response = None

        def capture_response(event):
            nonlocal captured_response
            if event.name == 'response_completed':
                captured_response = event.typed_event.msg

        # 临时订阅响应完成事件
        subtask.event_bus.on_event('response_completed', capture_response)

        try:
            # 运行子任务
            response = subtask.run(instruction)
            execution_time = time.time() - start_time

            # 直接提取LLM的响应内容，让主任务判断结果
            status = "completed"
            result_content = "No response content"

            # 优先使用捕获的响应，其次使用返回的响应
            target_response = captured_response or response

            if target_response:
                self.log.info(f"Using response: captured={captured_response is not None}, returned={response is not None}")

                # 如果是ChatMessage，直接获取内容
                if hasattr(target_response, 'content'):
                    content = target_response.content
                    if content and content.strip():
                        result_content = content.strip()
                        self.log.info(f"Successfully extracted content from ChatMessage: {content[:100]}...")
                    else:
                        result_content = "SubTask completed with empty ChatMessage content"

                # 如果是Response对象，获取其message
                elif hasattr(target_response, 'message'):
                    chat_message = target_response.message
                    self.log.info(f"Chat message type: {type(chat_message)}")

                    # 确保ChatMessage已经恢复了message内容
                    if hasattr(chat_message, 'message') and chat_message.message is None:
                        context = {'message_storage': subtask.message_storage}
                        chat_message.model_post_init(context)

                    # 获取content
                    content = chat_message.content
                    if content and content.strip():
                        result_content = content.strip()
                        self.log.info(f"Successfully extracted content from Response: {content[:100]}...")
                    else:
                        result_content = "SubTask completed with empty Response content"
                else:
                    result_content = f"SubTask target_response has unknown structure, type: {type(target_response)}"
            else:
                result_content = "SubTask: no captured or returned response"

            # 创建结果字典
            result = {
                "subtask_id": subtask.task_id,
                "status": status,
                "result": result_content,
                "execution_time": execution_time,
                "steps_count": len(subtask.steps)
            }

            # 发送SubTask完成事件
            self.parent_task.emit('subtask_completed',
                                 subtask_id=subtask.task_id,
                                 parent_task_id=self.parent_task.task_id,
                                 status=status,
                                 result=result_content,
                                 execution_time=execution_time,
                                 steps_count=len(subtask.steps))

            self.log.info(f"SubTask {subtask.task_id} completed",
                         parent_id=self.parent_task.task_id,
                         execution_time=execution_time,
                         steps=len(subtask.steps))

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.log.exception(f"SubTask {subtask.task_id} failed")

            result = {
                "subtask_id": subtask.task_id,
                "status": "completed",
                "result": f"SubTask execution failed with exception: {str(e)}",
                "execution_time": execution_time,
                "steps_count": len(subtask.steps)
            }

            # 发送SubTask完成事件（即使有异常也算完成，让主任务判断）
            self.parent_task.emit('subtask_completed',
                                 subtask_id=subtask.task_id,
                                 parent_task_id=self.parent_task.task_id,
                                 status="completed",
                                 result=f"SubTask execution failed with exception: {str(e)}",
                                 execution_time=execution_time,
                                 steps_count=len(subtask.steps))

            return result

        finally:
            # 手动移除事件监听器
            if 'response_completed' in subtask.event_bus._listeners:
                try:
                    subtask.event_bus._listeners['response_completed'].remove(capture_response)
                except ValueError:
                    pass  # 如果没找到也不要紧
            # 清理子任务引用
            self.subtasks.pop(subtask.task_id, None)

    def _inherit_context(self, subtask: 'Task'):
        """继承父任务的上下文"""
        # 简单实现：添加父任务的上下文摘要
        parent_context = self.parent_task.context_manager.get_messages()
        if parent_context:
            # 创建上下文摘要
            context_summary = f"[Inherited from parent task {self.parent_task.task_id}]\\n"
            context_summary += f"Parent task has {len(parent_context)} messages\\n"
            context_summary += "[End of inherited context]"

            # 添加到子任务上下文
            from ..llm import UserMessage
            from .chat import ChatMessage

            summary_message = ChatMessage(
                message=UserMessage(content=context_summary),
                id=f"inherited_{int(time.time())}"
            )
            subtask.context_manager.add_message(summary_message)
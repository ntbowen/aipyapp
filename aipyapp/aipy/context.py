#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum

from loguru import logger
from pydantic import BaseModel, Field

from ..llm import MessageRole, UserMessage
from .chat import ChatMessage, MessageStorage

class ContextStrategy(str, Enum):
    """上下文管理策略"""
    SLIDING_WINDOW = "sliding_window"      # 滑动窗口
    IMPORTANCE_FILTER = "importance_filter"  # 重要性过滤
    SUMMARY_COMPRESSION = "summary_compression"  # 摘要压缩
    HYBRID = "hybrid"                      # 混合策略

class ITokenEstimator(ABC):
    """Token估算器接口"""
    
    @abstractmethod
    def estimate(self, message: ChatMessage) -> int:
        """估算内容的token数量"""
        pass


class DefaultTokenEstimator(ITokenEstimator):
    """默认Token估算器实现"""
    
    def estimate(self, message: ChatMessage) -> int:
        content = message.content
        if isinstance(content, str):
            return len(content) // 4
        elif isinstance(content, list):
            total_length = 0
            for item in content:
                if item.get('type') == 'text':
                    total_length += len(item.get('text', ''))
            return total_length // 4
        return 0

class ContextConfig(BaseModel):
    """上下文管理配置"""
    max_tokens: int = Field(default=100000, description="最大token数")
    max_rounds: int = Field(default=10, description="最大对话轮数")
    auto_compress: bool = Field(default=True, description="是否自动压缩")
    strategy: ContextStrategy = Field(default=ContextStrategy.HYBRID, description="压缩策略")
    compression_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="压缩比例")
    importance_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性阈值")
    summary_max_length: int = Field(default=200, gt=0, description="摘要最大长度")
    preserve_system: bool = Field(default=True, description="是否保留系统消息")
    preserve_recent: int = Field(default=3, gt=0, description="保留最近几轮对话")

    def set_strategy(self, strategy: str) -> bool:
        try:
            # 尝试通过值创建枚举实例
            self.strategy = ContextStrategy(strategy)
            return True
        except ValueError:
            # 如果值不存在，尝试通过成员名称
            try:
                self.strategy = ContextStrategy[strategy.upper()]
                return True
            except KeyError:
                return False

class IContextStrategy(ABC):
    """上下文压缩策略接口"""
    def __init__(self, message_store: MessageStorage, config: ContextConfig, 
                 estimator: ITokenEstimator):
        self.message_store = message_store
        self.config = config
        self.estimator = estimator
        self.log = logger.bind(src=self.__class__.__name__)

    @abstractmethod
    def compress(self, context_data: 'ContextData') -> None:
        """压缩上下文数据（原地修改）"""
        pass

class SlidingWindowStrategy(IContextStrategy):
    """滑动窗口压缩策略"""
    
    def compress(self, context_data: 'ContextData') -> None:
        if context_data.total_tokens <= self.config.max_tokens:
            return
            
        original_count = len(context_data.messages)
        preserved_messages = []
        preserved_tokens = 0
        
        # 保留系统消息
        system_messages = [msg for msg in context_data.messages if msg.role == MessageRole.SYSTEM]
        for msg in system_messages:
            preserved_messages.append(msg)
            preserved_tokens += self.estimator.estimate(msg)
        
        # 保留最近的对话
        recent_messages = [msg for msg in context_data.messages if msg.role != MessageRole.SYSTEM]
        max_recent = self.config.preserve_recent * 2
        
        for msg in recent_messages[-max_recent:]:
            msg_tokens = self.estimator.estimate(msg)
            if preserved_tokens + msg_tokens <= self.config.max_tokens:
                preserved_messages.append(msg)
                preserved_tokens += msg_tokens
            else:
                break
        
        # 更新上下文数据
        context_data.messages = preserved_messages
        context_data.total_tokens = preserved_tokens
        
        self.log.info(f"Sliding window compression: {original_count} -> {len(preserved_messages)} messages, {preserved_tokens} tokens")

class ImportanceFilterStrategy(IContextStrategy):
    """重要性过滤压缩策略"""
    
    def compress(self, context_data: 'ContextData') -> None:
        if context_data.total_tokens <= self.config.max_tokens:
            return
            
        original_count = len(context_data.messages)
        messages = context_data.messages
        
        # 计算消息重要性分数
        scored_messages = []
        for i, msg in enumerate(messages):
            score = self._calculate_importance_score(msg, i, len(messages))
            scored_messages.append((score, msg))
        
        # 按重要性排序
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        preserved_messages = []
        preserved_tokens = 0
        
        for score, msg in scored_messages:
            msg_tokens = self.estimator.estimate(msg)
            if preserved_tokens + msg_tokens <= self.config.max_tokens:
                preserved_messages.append(msg)
                preserved_tokens += msg_tokens
            else:
                break
        
        # 按原始顺序重新排序
        preserved_messages.sort(key=lambda x: messages.index(x))
        
        # 更新上下文数据
        context_data.messages = preserved_messages
        context_data.total_tokens = preserved_tokens
        
        self.log.info(f"Importance filter compression: {original_count} -> {len(preserved_messages)} messages, {preserved_tokens} tokens")
    
    def _calculate_importance_score(self, message: ChatMessage, index: int, total: int) -> float:
        """计算消息重要性分数"""
        score = 0.0
        
        # 基于角色评分
        role = message.role
        if role == MessageRole.SYSTEM:
            score += 1.0
        elif role == MessageRole.USER:
            score += 0.8
        elif role == MessageRole.ASSISTANT:
            score += 0.6
        
        # 基于位置评分（越新越重要）
        recency = (index + 1) / total
        score += recency * 0.4
        
        # 基于内容长度评分
        content = message.content
        if isinstance(content, str):
            length_score = min(len(content) / 1000, 1.0)
            score += length_score * 0.2
        
        return score

class SummaryCompressionStrategy(IContextStrategy):
    """摘要压缩策略"""
    
    def compress(self, context_data: 'ContextData') -> None:
        if context_data.total_tokens <= self.config.max_tokens:
            return
            
        original_count = len(context_data.messages)
        preserved_messages: List[ChatMessage] = []
        preserved_tokens = 0
        
        # 保留系统消息
        system_messages = [msg for msg in context_data.messages if msg.role == MessageRole.SYSTEM]
        for msg in system_messages:
            preserved_messages.append(msg)
            preserved_tokens += self.estimator.estimate(msg)
        
        # 保留最近的对话
        recent_messages: List[ChatMessage] = [msg for msg in context_data.messages if msg.role != MessageRole.SYSTEM]
        max_recent = self.config.preserve_recent * 2
        
        # 分割消息
        if len(recent_messages) > max_recent:
            old_messages = recent_messages[:-max_recent]
            
            # 创建摘要消息
            summary_content = self._create_summary(old_messages)
            summary_msg = UserMessage(content=f"对话历史摘要：{summary_content}")
            
            preserved_messages.append(self.message_store.store(summary_msg))
            preserved_tokens += self.estimator.estimate(summary_msg.content)
        
        # 添加新消息
        for msg in recent_messages[-max_recent:]:
            msg_tokens = self.estimator.estimate(msg)
            if preserved_tokens + msg_tokens <= self.config.max_tokens:
                preserved_messages.append(msg)
                preserved_tokens += msg_tokens
            else:
                break
        
        # 更新上下文数据
        context_data.messages = preserved_messages
        context_data.total_tokens = preserved_tokens
        
        self.log.info(f"Summary compression: {original_count} -> {len(preserved_messages)} messages, {preserved_tokens} tokens")
    
    def _create_summary(self, messages: List[ChatMessage]) -> str:
        """创建消息摘要"""
        summary_parts = []
        
        for msg in messages:
            role = msg.role
            content = msg.content
            
            if isinstance(content, str):
                content_preview = content[:100] + ('...' if len(content) > 100 else '')
                summary_parts.append(f"{role}: {content_preview}")
        
        summary = " | ".join(summary_parts)
        return summary[:self.config.summary_max_length]

class HybridStrategy(IContextStrategy):
    """混合压缩策略"""
    
    def __init__(self, message_store: MessageStorage, config: ContextConfig, 
                 estimator: ITokenEstimator):
        super().__init__(message_store, config, estimator)
        self.sliding_window = SlidingWindowStrategy(message_store, config, estimator)
        self.summary_compression = SummaryCompressionStrategy(message_store, config, estimator)
    
    def compress(self, context_data: 'ContextData') -> None:
        if context_data.total_tokens <= self.config.max_tokens:
            return
            
        # 首先尝试滑动窗口
        self.sliding_window.compress(context_data)
        
        # 如果仍然超出限制，使用摘要压缩
        if context_data.total_tokens > self.config.max_tokens:
            self.summary_compression.compress(context_data)

class ContextStrategyFactory:
    """上下文策略工厂类"""
    
    _strategies: Dict[ContextStrategy, type] = {
        ContextStrategy.SLIDING_WINDOW: SlidingWindowStrategy,
        ContextStrategy.IMPORTANCE_FILTER: ImportanceFilterStrategy,
        ContextStrategy.SUMMARY_COMPRESSION: SummaryCompressionStrategy,
        ContextStrategy.HYBRID: HybridStrategy,
    }
    
    @classmethod
    def create(cls, strategy_type: ContextStrategy, message_store: MessageStorage,
               config: ContextConfig, estimator: ITokenEstimator) -> IContextStrategy:
        """创建指定类型的压缩策略"""
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
        
        strategy_class = cls._strategies[strategy_type]
        return strategy_class(message_store, config, estimator)
    
    @classmethod
    def register_strategy(cls, strategy_type: ContextStrategy, strategy_class: type):
        """注册新的策略类型"""
        if not issubclass(strategy_class, IContextStrategy):
            raise TypeError("Strategy class must implement IContextStrategy interface")
        cls._strategies[strategy_type] = strategy_class

class MessageCompressor:
    """消息压缩器 - 重构为使用策略模式"""
    
    def __init__(self, message_store: MessageStorage, config: ContextConfig, 
                 estimator: ITokenEstimator = None):
        self.config = config
        self.message_store = message_store
        self.estimator = estimator or DefaultTokenEstimator()
        self.strategy = ContextStrategyFactory.create(config.strategy, message_store, config, self.estimator)
        self.log = logger.bind(src='message_compressor')
    
    def compress_context(self, context_data: 'ContextData') -> None:
        """压缩上下文数据"""
        self.strategy.compress(context_data)
    
    def update_strategy(self, new_strategy: ContextStrategy):
        """更新压缩策略"""
        self.strategy = ContextStrategyFactory.create(new_strategy, self.message_store, self.config, self.estimator)
        self.log.info(f"Strategy updated to: {new_strategy.value}")
    
    def update_config(self, new_config: ContextConfig):
        """更新配置并重新创建策略"""
        self.config = new_config
        self.strategy = ContextStrategyFactory.create(new_config.strategy, self.message_store, new_config, self.estimator)
        self.log.info(f"Config updated: {new_config.strategy.value}")
    
    def estimate_message_tokens(self, message: ChatMessage) -> int:
        """估算消息token数（向后兼容，建议使用estimate_single_message_tokens）"""
        return self.estimator.estimate(message)

class ContextData(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    total_tokens: int = 0
    
    def __len__(self):
        return len(self.messages)
    
class ContextManager:
    """上下文管理器"""
    
    def __init__(self, message_store: MessageStorage, data: ContextData, 
                 config: Union[dict, ContextConfig, None] = None):
        if isinstance(config, dict):
            self.config = ContextConfig(**config)
        elif isinstance(config, ContextConfig):
            self.config = config
        else:
            self.config = ContextConfig()
        
        self.message_store = message_store
        self.compressor = MessageCompressor(message_store, self.config)
        self.log = logger.bind(src='context_manager')
        
        self.data = data
        self._last_compression_time = 0
        
    @property
    def total_tokens(self):
        return self.data.total_tokens
    
    @property
    def messages(self):
        return self.data.messages
    
    def add_chat(self, user_message: ChatMessage, llm_message: ChatMessage):
        """添加用户-LLM对话（推荐使用）"""
        self.data.messages.extend([user_message, llm_message])
        
        # 尝试从LLM消息提取准确的总token数
        context_total = llm_message.total_tokens
        if context_total:
            self.data.total_tokens = context_total
            self.log.info(f"Updated context tokens from LLM usage: {self.data.total_tokens}")
        else:
            # 兜底：累加两条消息的估算
            user_tokens = self.compressor.estimate_message_tokens(user_message)
            llm_tokens = self.compressor.estimate_message_tokens(llm_message)
            self.data.total_tokens += user_tokens + llm_tokens
            self.log.info(f"Estimated tokens for new chat: +{user_tokens + llm_tokens}, total: {self.data.total_tokens}")
    
    def add_message(self, message: ChatMessage):
        """添加单独消息（系统消息、反馈消息等）"""
        self.data.messages.append(message)
        
        # 检查是否是包含总token数的Assistant消息
        context_total = message.total_tokens
        if context_total:
            self.data.total_tokens = context_total
            self.log.info(f"Updated context tokens from LLM usage: {self.data.total_tokens}")
        else:
            # 普通消息，累加估算的token
            message_tokens = self.compressor.estimate_message_tokens(message)
            self.data.total_tokens += message_tokens
            self.log.info(f"Added single message: +{message_tokens} tokens, total: {self.data.total_tokens}, id: {message.id}")
    
    def get_messages(self, force_compress: bool = False) -> List[ChatMessage]:
        """获取压缩后的消息列表"""
        current_time = time.time()
        
        # 检查是否需要压缩
        if force_compress or self.config.auto_compress:
            should_compress = (
                force_compress or
                self.total_tokens > self.config.max_tokens or
                len(self.data.messages) > self.config.max_rounds * 2 or
                (current_time - self._last_compression_time) > 300  # 5分钟强制压缩
            )
            
            if should_compress:
                self.compress()
        
        return self.messages.copy()
    
    def compress(self):
        """压缩消息"""
        if not self.data.messages:
            return
        
        original_count = len(self.data.messages)
        original_tokens = self.total_tokens
        
        # 执行压缩
        self.compressor.compress_context(self.data)
        
        self._last_compression_time = time.time()
        
        self.log.info(
            f"Context compressed: {original_count}->{len(self.data.messages)} messages, "
            f"{original_tokens}->{self.data.total_tokens} tokens"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取上下文统计信息"""
        return {
            'message_count': len(self.data.messages),
            'total_tokens': self.data.total_tokens,
            'last_compression': self._last_compression_time
        }
    
    def clear(self):
        """清理消息缓存，只保留最初的两条消息和最后一条消息"""
        messages = self.data.messages
        if not messages or len(messages) <= 2:
            return
            
        if messages[0].role == MessageRole.SYSTEM:
            if len(messages) > 3:
                messages[2] = messages[-1]
                del messages[3:]
        else:
            messages[1] = messages[-1]
            del messages[2:]
        
        self._last_compression_time = 0
        self.data.total_tokens = sum(self.compressor.estimate_message_tokens(msg) for msg in messages)
        self.log.info(f"Context cleaned: {len(self.data.messages)} messages, {self.data.total_tokens} tokens")

    def rebuild(self, messages: List[ChatMessage]):
        """重建消息缓存"""
        self.data.messages.clear()
        self.data.total_tokens = 0
        
        for message in messages:
            self.add_message(message)
    
    def update_config(self, config: ContextConfig):
        """更新配置"""
        self.config = config
        self.compressor.update_config(config)
        self.log.info(f"Context config updated: {config.strategy.value}")
    
    def delete_messages_by_ids(self, message_ids: List[str]) -> Tuple[int, int]:
        """
        按ID删除消息，带安全保护
        
        Args:
            message_ids: 要删除的消息ID列表
            
        Returns:
            Tuple[删除数量, 节省的token数]
        """
        if not message_ids:
            return 0, 0
            
        messages = self.data.messages
        if len(messages) <= 2:  # 至少保留系统消息和第一条用户消息
            return 0, 0
        
        # 计算需要保护的消息索引
        protected_indices = set()
        
        # 保护系统消息
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.SYSTEM:
                protected_indices.add(i)
        
        # 保护第一条用户消息（通常是任务指令）
        first_user_found = False
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.USER and not first_user_found:
                protected_indices.add(i)
                first_user_found = True
                break
        
        # 收集要删除的消息
        to_delete = []
        tokens_saved = 0
        
        for i, msg in enumerate(messages):
            if msg.id in message_ids and i not in protected_indices:
                to_delete.append(i)
                tokens_saved += self.compressor.estimate_message_tokens(msg)
        
        # 从后向前删除，避免索引变化
        for i in sorted(to_delete, reverse=True):
            del messages[i]
        
        # 更新token计数
        self.data.total_tokens = sum(self.compressor.estimate_message_tokens(msg) for msg in messages)
        
        deleted_count = len(to_delete)
        if deleted_count > 0:
            self.log.info(f"Deleted {deleted_count} messages by ID, saved {tokens_saved} tokens")
        
        return deleted_count, tokens_saved
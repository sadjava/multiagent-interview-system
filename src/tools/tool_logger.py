"""
Tool Logger - централизованное логирование использования инструментов.
Отслеживает все вызовы tools и формирует отчёт для internal_thought.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import threading


@dataclass
class ToolCall:
    """Запись о вызове инструмента"""
    tool_name: str
    input_data: str
    output_data: str
    timestamp: datetime
    success: bool
    duration_ms: float = 0.0


class ToolLogger:
    """
    Централизованный логгер для отслеживания использования инструментов.
    Thread-safe для параллельного выполнения агентов.
    """
    
    def __init__(self):
        self._calls: List[ToolCall] = []
        self._lock = threading.Lock()
        self._current_agent: str = ""
    
    def set_agent(self, agent_name: str):
        """Устанавливает текущего агента для логирования"""
        self._current_agent = agent_name
    
    def log_call(
        self,
        tool_name: str,
        input_data: str,
        output_data: str,
        success: bool = True,
        duration_ms: float = 0.0
    ):
        """
        Логирует вызов инструмента.
        
        Args:
            tool_name: Название инструмента
            input_data: Входные данные (краткое описание)
            output_data: Результат (краткое описание)
            success: Успешен ли вызов
            duration_ms: Время выполнения в миллисекундах
        """
        with self._lock:
            call = ToolCall(
                tool_name=f"{self._current_agent}.{tool_name}" if self._current_agent else tool_name,
                input_data=input_data[:200],  # Обрезаем длинные данные
                output_data=output_data[:200],
                timestamp=datetime.now(),
                success=success,
                duration_ms=duration_ms
            )
            self._calls.append(call)
            
            # Выводим в консоль
            status = "✓" if success else "✗"
            print(f"[Tool] {status} {call.tool_name}: {input_data[:50]}... -> {output_data[:50]}...")
    
    def get_summary(self) -> str:
        """
        Возвращает краткое резюме использованных инструментов для internal_thought.
        
        Returns:
            Строка с описанием использованных инструментов
        """
        with self._lock:
            if not self._calls:
                return ""
            
            summaries = []
            for call in self._calls:
                status = "успешно" if call.success else "ошибка"
                summaries.append(f"[{call.tool_name}]: {call.input_data[:50]}... ({status})")
            
            return "Использованные инструменты: " + "; ".join(summaries)
    
    def get_calls(self) -> List[ToolCall]:
        """Возвращает список всех вызовов"""
        with self._lock:
            return list(self._calls)
    
    def clear(self):
        """Очищает лог вызовов"""
        with self._lock:
            self._calls.clear()
    
    def get_thought_addon(self) -> str:
        """
        Возвращает дополнение к internal_thought с информацией об инструментах.
        
        Returns:
            Строка для добавления к internal_thought агента
        """
        with self._lock:
            if not self._calls:
                return ""
            
            tools_used = [call.tool_name.split(".")[-1] for call in self._calls]
            unique_tools = list(set(tools_used))
            
            return f" [Инструменты: {', '.join(unique_tools)}]"


# Глобальный инстанс логгера
_logger_instance: Optional[ToolLogger] = None


def get_tool_logger() -> ToolLogger:
    """Возвращает глобальный инстанс логгера"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ToolLogger()
    return _logger_instance


def log_tool(tool_name: str, input_data: str, output_data: str, success: bool = True):
    """Удобная функция для быстрого логирования"""
    get_tool_logger().log_call(tool_name, input_data, output_data, success)


"""
LangGraph - сборка графа состояний для интервью.
The Cognitive Council - главный граф системы.

ПАЙПЛАЙН:
1. start_interview() -> Planner + Voice генерируют первый вопрос
2. Кандидат отвечает
3. Router -> определяет интент
4. answer -> Skeptic + Empath (параллельно) -> Planner -> Voice
5. question/off_topic -> Planner -> Voice
6. stop -> Reporter
"""

import os
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END

from .state import InterviewState, create_initial_state
from .agents.router import router_node
from .agents.skeptic import skeptic_node
from .agents.empath import empath_node
from .agents.planner import planner_node, create_interview_plan
from .agents.voice import voice_node
from .agents.reporter import reporter_node


def parallel_analysis_node(state: InterviewState) -> Dict[str, Any]:
    """Узел параллельного анализа - Skeptic и Empath одновременно."""
    print("[ParallelAnalysis] Запуск Skeptic и Empath параллельно...")
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_skeptic = executor.submit(skeptic_node, state)
        future_empath = executor.submit(empath_node, state)
        
        futures = {
            future_skeptic: "skeptic",
            future_empath: "empath"
        }
        
        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                results[agent_name] = result
                print(f"[ParallelAnalysis] {agent_name.capitalize()} завершен")
            except Exception as e:
                print(f"[ParallelAnalysis] Ошибка в {agent_name}: {e}")
                results[agent_name] = {}
    
    merged_result = {}
    if "skeptic" in results:
        merged_result.update(results["skeptic"])
    if "empath" in results:
        merged_result.update(results["empath"])
    
    return merged_result


def build_interview_graph() -> StateGraph:
    """Строит граф состояний для проведения интервью."""
    
    workflow = StateGraph(InterviewState)
    
    # Узлы
    workflow.add_node("router", router_node)
    workflow.add_node("parallel_analysis", parallel_analysis_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("voice", voice_node)
    workflow.add_node("reporter", reporter_node)
    
    # Точка входа — всегда Router (кроме первого хода)
    workflow.set_entry_point("router")
    
    # Переходы от Router
    workflow.add_conditional_edges(
        "router",
        route_from_router,
        {
            "analyze": "parallel_analysis",
            "quick_respond": "planner",
            "first_turn": "planner",  # Первый ход — сразу в planner
            "end": "reporter"
        }
    )
    
    workflow.add_edge("parallel_analysis", "planner")
    
    workflow.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "respond": "voice",
            "end": "reporter"
        }
    )
    
    workflow.add_edge("voice", END)
    workflow.add_edge("reporter", END)
    
    return workflow.compile()


def route_from_router(state: InterviewState) -> str:
    """Определяет следующий узел после Router."""
    # Первый ход — сразу в planner (без анализа)
    if state.get("turn_id", 0) == 0:
        return "first_turn"
    
    next_step = state.get("next_step", "analyze")
    
    if next_step == "end":
        return "end"
    elif next_step == "analyze":
        return "analyze"
    else:
        return "quick_respond"


def route_from_planner(state: InterviewState) -> str:
    """Определяет следующий узел после Planner."""
    if state.get("should_end", False):
        return "end"
    return "respond"


class InterviewCoach:
    """
    Главный класс для проведения интервью.
    
    Простой пайплайн:
    1. start_interview() — инициализация + первый вопрос через Planner+Voice
    2. process_message() — обработка ответов кандидата
    """
    
    def __init__(self):
        self.graph = build_interview_graph()
        self.state: InterviewState = None
        self.is_active = False
        self._last_agent_message = ""  # Для правильного логирования
    
    def start_interview(
        self,
        name: str,
        role: str,
        grade: str,
        experience: str
    ) -> str:
        """
        Инициализирует интервью и генерирует первое сообщение через Planner+Voice.
        
        Returns:
            Первое сообщение интервьюера
        """
        # Создаем начальное состояние
        self.state = create_initial_state(name, role, grade, experience)
        
        # Создаем план интервью
        plan_result = create_interview_plan(self.state)
        self.state.update(plan_result)
        
        # turn_id = 0 означает, что это первый ход
        self.state["turn_id"] = 0
        self.state["current_user_message"] = ""  # Пустое — кандидат ещё не отвечал
        self.state["turns_log"] = []
        
        # Запускаем граф — он пройдёт Router -> Planner -> Voice
        result = self.graph.invoke(self.state)
        self.state = result
        
        self.is_active = True
        
        # Получаем первое сообщение агента
        first_message = result.get("current_response", "")
        if not first_message and result.get("messages"):
            for msg in reversed(result["messages"]):
                if msg["role"] == "assistant":
                    first_message = msg["content"]
                    break
        
        # Сохраняем для логирования
        self._last_agent_message = first_message
        
        return first_message
    
    def process_message(self, user_message: str) -> str:
        """
        Обрабатывает сообщение кандидата и возвращает ответ интервьюера.
        
        Логирование:
        - Каждый turn содержит: предыдущий вопрос агента + ответ кандидата + мысли агентов
        """
        if not self.is_active:
            return "Интервью не активно. Пожалуйста, начните новое интервью."
        
        # Сохраняем предыдущий вопрос для лога
        previous_question = self._last_agent_message
        current_turn_id = self.state["turn_id"]
        
        # Обновляем состояние
        self.state["current_user_message"] = user_message
        
        # Сбрасываем временные поля
        self.state["skeptic_analysis"] = ""
        self.state["empath_analysis"] = ""
        self.state["planner_directive"] = ""
        self.state["internal_debate"] = ""
        self.state["hallucination_detected"] = False
        self.state["router_thought"] = ""
        self.state["skeptic_thought"] = ""
        self.state["empath_thought"] = ""
        self.state["planner_thought"] = ""
        self.state["voice_thought"] = ""
        self.state["_move_to_next_topic"] = False
        
        # Запускаем граф
        result = self.graph.invoke(self.state)
        self.state = result
        
        # Получаем ответ агента
        agent_response = result.get("current_response", "")
        if not agent_response and result.get("messages"):
            for msg in reversed(result["messages"]):
                if msg["role"] == "assistant":
                    agent_response = msg["content"]
                    break
        
        # Проверяем, это reporter (финальный отчёт)?
        if result.get("should_end"):
            self.is_active = False
            # Для reporter используем final_report_string
            if result.get("final_report_string"):
                agent_response = result["final_report_string"]
        
        # Записываем в лог: предыдущий вопрос + ответ кандидата + мысли
        turn_log = {
            "turn_id": current_turn_id,
            "agent_visible_message": previous_question,
            "user_message": user_message,
            "internal_thoughts": result.get("internal_debate", "")
        }
        
        # Обновляем turns_log
        self.state["turns_log"] = self.state.get("turns_log", []) + [turn_log]
        
        # Сохраняем текущий ответ для следующего хода
        self._last_agent_message = agent_response
        
        return agent_response
    
    def get_state(self) -> InterviewState:
        """Возвращает текущее состояние"""
        return self.state
    
    def get_turns_log(self) -> list:
        """Возвращает лог ходов"""
        return self.state.get("turns_log", []) if self.state else []
    
    def get_final_feedback(self) -> dict:
        """Возвращает финальный фидбэк"""
        return self.state.get("final_feedback") if self.state else None
    
    def is_interview_active(self) -> bool:
        """Проверяет, активно ли интервью"""
        return self.is_active
    
    def get_internal_thoughts(self) -> str:
        """Возвращает internal_debate текущего хода"""
        return self.state.get("internal_debate", "") if self.state else ""
    
    def get_agent_thoughts(self) -> Dict[str, str]:
        """Возвращает internal_thought от каждого агента"""
        if not self.state:
            return {}
        return {
            "router": self.state.get("router_thought", ""),
            "skeptic": self.state.get("skeptic_thought", ""),
            "empath": self.state.get("empath_thought", ""),
            "planner": self.state.get("planner_thought", ""),
            "voice": self.state.get("voice_thought", "")
        }
    
    def export_session(self) -> dict:
        """Экспортирует сессию в формате для JSON-лога."""
        if not self.state:
            return {}
        
        return {
            "participant_name": "Садреддинов Джавид Ханбаба оглы",
            "turns": self.state.get("turns_log", []),
            # final_feedback как строка (final_report_string)
            "final_feedback": self.state.get("final_report_string", "")
        }

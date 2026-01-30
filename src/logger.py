"""
Interview Logger - система логирования для сохранения сессий.
Сохраняет логи в формате JSON согласно ТЗ.
"""

import json
import os
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from .state import InterviewState, TurnLog, FinalFeedback


class InterviewLogger:
    """
    Логгер для сохранения сессий интервью.
    Формат соответствует требованиям ТЗ.
    """
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self.current_session: dict = {}
        self.session_file: Optional[Path] = None
    
    def start_session(self, participant_name: str, scenario_id: int) -> str:
        """Начинает новую сессию логирования"""
        filename = f"interview_log_{scenario_id}.json"
        self.session_file = self.logs_dir / filename
        
        self.current_session = {
            "participant_name": participant_name,
            "turns": [],
            "final_feedback": None
        }
        
        self._save()
        return str(self.session_file)
    
    def log_turn(
        self,
        turn_id: int,
        agent_visible_message: str,
        user_message: str,
        internal_thoughts: str
    ) -> None:
        """Логирует один ход интервью"""
        turn = {
            "turn_id": turn_id,
            "agent_visible_message": agent_visible_message,
            "user_message": user_message,
            "internal_thoughts": internal_thoughts
        }
        self.current_session["turns"].append(turn)
        self._save()
    
    def log_final_feedback(self, feedback: dict) -> None:
        """Сохраняет финальный фидбэк"""
        self.current_session["final_feedback"] = feedback
        self._save()
    
    def _save(self) -> None:
        """Сохраняет текущую сессию в файл"""
        if self.session_file:
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(self.current_session, f, ensure_ascii=False, indent=2)
    
    def get_session_path(self) -> Optional[str]:
        """Возвращает путь к файлу текущей сессии"""
        return str(self.session_file) if self.session_file else None
    
    def format_internal_thoughts(
        self,
        skeptic: str,
        empath: str,
        planner: str
    ) -> str:
        """
        Форматирует мысли агентов в единую строку.
        Формат: [Skeptic]: ... | [Empath]: ... | [Planner]: ...
        """
        parts = []
        if skeptic:
            parts.append(f"[Skeptic]: {skeptic}")
        if empath:
            parts.append(f"[Empath]: {empath}")
        if planner:
            parts.append(f"[Planner]: {planner}")
        return " | ".join(parts)


# Глобальный экземпляр логгера
interview_logger = InterviewLogger()


def export_state_to_log(state: InterviewState) -> dict:
    """
    Экспортирует состояние в формат для JSON-лога.
    Используется для финального сохранения.
    """
    return {
        "participant_name": state["metadata"]["name"],
        "metadata": {
            "role": state["metadata"]["role"],
            "target_grade": state["metadata"]["target_grade"],
            "experience": state["metadata"]["experience"]
        },
        "turns": state["turns_log"],
        "final_feedback": state.get("final_feedback")
    }

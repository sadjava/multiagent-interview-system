"""
Strategic Planner - агент-агрегатор и дирижер системы.
Собирает мнения Skeptic и Empath, обновляет план и дает директивы Voice.
Генерирует план интервью на основе данных кандидата.
"""

import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, Optional

from ..state import InterviewState, PlannerOutput


# ============================================================
# Модели для генерации плана интервью
# ============================================================

class PlanTopic(BaseModel):
    """Один элемент плана интервью"""
    topic: str = Field(description="Конкретная тема для проверки")
    difficulty: Literal["easy", "medium", "hard", "expert"] = Field(
        description="Сложность темы"
    )
    rationale: str = Field(description="Почему эта тема важна")


class InterviewPlanOutput(BaseModel):
    """Structured output для генерации плана интервью"""
    topics: List[PlanTopic] = Field(
        description="Список из 6-8 конкретных тем, от базовых к сложным"
    )
    internal_thought: str = Field(
        description="Краткое обоснование выбора тем"
    )


class QuickPlannerOutput(BaseModel):
    """Structured output для быстрого планирования (question/off_topic)"""
    directive: str = Field(description="Инструкция для Voice")
    internal_thought: str = Field(description="Причина решения")


# ============================================================
# Промпты
# ============================================================

INTERVIEW_PLAN_PROMPT = """Ты - опытный технический интервьюер. Сгенерируй план интервью.

ДАННЫЕ КАНДИДАТА:
- Позиция: {role}
- Заявленный грейд: {grade}
- Опыт: {experience}

ТРЕБОВАНИЯ К ПЛАНУ:
1. Сгенерируй 6-8 КОНКРЕТНЫХ тем для проверки
2. Темы должны быть релевантны позиции и опыту кандидата
3. Начни с базовых тем, постепенно усложняя
4. НЕ используй общие фразы типа "общие вопросы" - будь конкретен
5. Учитывай упомянутые технологии в опыте

ПРИМЕРЫ ХОРОШИХ ТЕМ:
- "Django ORM: QuerySet API, select_related, prefetch_related"
- "SQL: JOIN типы, индексы, оптимизация"
- "Python: генераторы, декораторы, контекстные менеджеры"
"""


PLANNER_PROMPT = """Ты - Strategic Planner. Управляешь ходом интервью.

КАНДИДАТ: {name}, {role}, {grade}
ХОД: {turn_id}
ТЕКУЩАЯ ТЕМА: {current_topic}

ПЛАН: {plan_status}

SKEPTIC: {skeptic_analysis}
EMPATH: {empath_analysis}

ИНТЕНТ: {user_intent}
ОТВЕТ: {user_message}

ГАЛЛЮЦИНАЦИЯ: {hallucination_detected}
ПРАВИЛЬНЫЙ ФАКТ: {correct_answer}

ПРАВИЛА:
1. После каждого ответа — ПЕРЕХОДИ К СЛЕДУЮЩЕМУ ВОПРОСУ (по той же или новой теме)
2. Не застревай на одной теме дольше 2-3 вопросов
3. НЕ объясняй правильные ответы — это собеседование!
4. Двигайся по плану — охвати разные темы

🚨 КРИТИЧЕСКИ ВАЖНО - ГАЛЛЮЦИНАЦИИ:
- ГАЛЛЮЦИНАЦИИ (ложные факты) — это КРИТИЧЕСКОЕ нарушение! Реагируй ЖЁСТКО!
- Оффтопик (про погоду) — просто верни к теме
- Галлюцинация (выдуманные факты) — ОБЯЗАТЕЛЬНО помечай и кратко спори!
- НЕ путай: плохой ответ ≠ оффтопик ≠ галлюцинация

КОГДА МЕНЯТЬ ТЕМУ:
- Получили ответ (любой) → можно переходить к следующей теме
- 2+ вопроса по одной теме → точно переходи дальше
- Галлюцинация → отметь как критическую проблему, кратко уточни, затем переходи дальше

ДИРЕКТИВА ДЛЯ VOICE:
- Кратко (1 предложение): что спросить следующее
- НЕ объяснять, НЕ учить
- Если галлюцинация → КРАТКО спори/уточни: "Не уверен, что X существует, откуда информация?" или "X не существует, уточните пожалуйста"
- НЕ объясняй правильный ответ, только уточни источник или факт

ОТВЕТ:
- topic_score (0-10 или null)
- next_action: continue/next_topic/finish
- difficulty_change: increase/decrease/keep
- new_protocol: standard/rescue/speedrun/stress_test
- directive: что спросить (кратко!)
- internal_thought: 1 предложение
"""


QUICK_PLANNER_PROMPT = """Быстрое решение для интервью.

ХОД: {turn_id}
ИНТЕНТ: {user_intent}
СООБЩЕНИЕ: {user_message}
ТЕМА: {current_topic}

ДЕЙСТВИЕ:
- question → кратко ответь + задай следующий вопрос
- off_topic → верни к теме

ОТВЕТ:
- directive: что делать (кратко)
- internal_thought: причина
"""


def get_planner_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
        temperature=0.3,
        reasoning_effort="low",
        max_completion_tokens=2000
    ).with_structured_output(PlannerOutput)


def get_quick_planner_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
        temperature=0.3,
        reasoning_effort="low",
        max_completion_tokens=2000
    ).with_structured_output(QuickPlannerOutput)


def get_plan_generator_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.4,
        reasoning_effort="low",
        max_completion_tokens=4000  # План больше
    ).with_structured_output(InterviewPlanOutput)


def create_interview_plan(state: InterviewState) -> Dict[str, Any]:
    """Генерирует план интервью на основе данных кандидата."""
    role = state["metadata"]["role"]
    grade = state["metadata"]["target_grade"]
    experience = state["metadata"]["experience"]
    
    prompt = ChatPromptTemplate.from_template(INTERVIEW_PLAN_PROMPT)
    llm = get_plan_generator_llm()
    chain = prompt | llm
    
    try:
        print(f"[Planner] Генерация плана интервью для {role} ({grade})...")
        
        result: InterviewPlanOutput = chain.invoke({
            "role": role,
            "grade": grade,
            "experience": experience
        })
        
        interview_plan = []
        for i, topic in enumerate(result.topics[:8]):
            interview_plan.append({
                "id": i + 1,
                "topic": topic.topic,
                "difficulty": topic.difficulty,
                "rationale": topic.rationale,
                "status": "pending",
                "score": None,
                "feedback": "",
                "correct_answer": None,
                "weak_answers": 0  # Счётчик слабых ответов на этой теме
            })
        
        print(f"[Planner] План создан: {len(interview_plan)} тем")
        
        return {
            "interview_plan": interview_plan,
            "current_topic_index": 0,
            "planner_thought": result.internal_thought
        }
        
    except Exception as e:
        print(f"[Planner] Ошибка генерации плана: {e}")
        
        fallback_plan = [
            {
                "id": 1,
                "topic": f"Базовые навыки для {role}",
                "difficulty": "easy" if "junior" in grade.lower() else "medium",
                "rationale": "Проверка фундаментальных знаний",
                "status": "pending",
                "score": None,
                "feedback": "",
                "correct_answer": None,
                "weak_answers": 0
            },
            {
                "id": 2,
                "topic": f"Практический опыт: {experience[:50]}...",
                "difficulty": "medium",
                "rationale": "Проверка заявленного опыта",
                "status": "pending",
                "score": None,
                "feedback": "",
                "correct_answer": None,
                "weak_answers": 0
            }
        ]
        
        return {
            "interview_plan": fallback_plan,
            "current_topic_index": 0,
            "planner_thought": f"Fallback план: {str(e)}"
        }


def planner_node(state: InterviewState) -> Dict[str, Any]:
    """
    Узел Планировщика - агрегирует анализ и управляет планом.
    Также используется для первого хода (генерация первого вопроса).
    """
    user_intent = state.get("user_intent", "answer")
    turn_id = state.get("turn_id", 0)
    
    print(f"[Planner] Ход {turn_id}, интент: {user_intent}")
    
    # Первый ход — нужно начать интервью
    if turn_id == 0:
        return first_turn_plan(state)
    
    # Быстрый путь для question/off_topic
    if user_intent in ["question", "off_topic"]:
        return quick_plan(state)
    
    # Полный анализ для answer
    return full_plan(state)


def first_turn_plan(state: InterviewState) -> Dict[str, Any]:
    """Планирование первого хода — приветствие + первый вопрос"""
    
    current_topic = "Общие вопросы"
    if state["interview_plan"]:
        current_topic = state["interview_plan"][0]["topic"]
        state["interview_plan"][0]["status"] = "in_progress"
    
    directive = (
        f"Поприветствуй кандидата {state['metadata']['name']} (1 предложение). "
        f"Упомяни опыт: {state['metadata']['experience'][:50]}. "
        f"Задай ОДИН простой вопрос по теме: {current_topic}."
    )
    
    internal_thought = f"Начало интервью. Первая тема: {current_topic}"
    
    return {
        "planner_directive": directive,
        "planner_thought": internal_thought,
        "internal_debate": f"[Planner]: {internal_thought}",
        "next_step": "respond"
    }


def quick_plan(state: InterviewState) -> Dict[str, Any]:
    """Быстрое планирование для question/off_topic"""
    
    user_intent = state.get("user_intent", "answer")
    user_message = state["current_user_message"]
    turn_id = state.get("turn_id", 0)
    
    current_topic = "Текущая тема"
    if state["interview_plan"] and state["current_topic_index"] < len(state["interview_plan"]):
        current_topic = state["interview_plan"][state["current_topic_index"]]["topic"]
    
    prompt = ChatPromptTemplate.from_template(QUICK_PLANNER_PROMPT)
    llm = get_quick_planner_llm()
    chain = prompt | llm
    
    try:
        result: QuickPlannerOutput = chain.invoke({
            "turn_id": turn_id,
            "user_intent": user_intent,
            "user_message": user_message,
            "current_topic": current_topic
        })
        
        router_thought = state.get("router_thought", "")
        internal_debate = f"[Router]: {router_thought}\n[Planner]: {result.internal_thought}"
        
        return {
            "planner_directive": result.directive,
            "planner_thought": result.internal_thought,
            "internal_debate": internal_debate,
            "next_step": "respond"
        }
        
    except Exception as e:
        print(f"[Planner] Quick plan error: {e}")
        return {
            "planner_directive": "Продолжай интервью",
            "planner_thought": f"Ошибка: {str(e)}",
            "internal_debate": f"[Planner]: Ошибка: {str(e)}",
            "next_step": "respond"
        }


def full_plan(state: InterviewState) -> Dict[str, Any]:
    """Полное планирование с анализом Skeptic и Empath"""
    
    plan_status = format_plan_status(state["interview_plan"])
    
    # Текущая тема
    current_topic = "Нет активной темы"
    current_idx = state["current_topic_index"]
    
    if state["interview_plan"] and current_idx < len(state["interview_plan"]):
        topic = state["interview_plan"][current_idx]
        current_topic = topic["topic"]
    
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    llm = get_planner_llm()
    chain = prompt | llm
    
    try:
        result: PlannerOutput = chain.invoke({
            "name": state["metadata"]["name"],
            "role": state["metadata"]["role"],
            "grade": state["metadata"]["target_grade"],
            "plan_status": plan_status,
            "current_topic": current_topic,
            "turn_id": state["turn_id"],
            "skeptic_analysis": state.get("skeptic_analysis", "-"),
            "empath_analysis": state.get("empath_analysis", "-"),
            "user_intent": state["user_intent"],
            "user_message": state["current_user_message"],
            "hallucination_detected": "ДА" if state.get("hallucination_detected") else "НЕТ",
            "correct_answer": state.get("_skeptic_correct_answer", "-")
        })
        
        topic_score = result.topic_score
        next_action = result.next_action
        directive = result.directive
        internal_thought = result.internal_thought
        
        # Обновляем план
        updated_plan = state["interview_plan"].copy()
        new_topic_index = current_idx
        move_to_next = False
        
        if current_idx < len(updated_plan):
            # Записываем оценку
            if topic_score is not None:
                updated_plan[current_idx]["score"] = topic_score
            
            # Увеличиваем счётчик вопросов по теме
            updated_plan[current_idx]["questions_asked"] = updated_plan[current_idx].get("questions_asked", 0) + 1
            
            # После 2 вопросов по теме — переходим дальше (чтобы охватить больше тем)
            if updated_plan[current_idx].get("questions_asked", 0) >= 2:
                move_to_next = True
        
        # Переход к следующей теме
        if next_action == "next_topic" or move_to_next:
            if current_idx < len(updated_plan):
                updated_plan[current_idx]["status"] = "completed"
            new_topic_index = find_next_pending_topic(updated_plan)
            
            if new_topic_index < len(updated_plan):
                updated_plan[new_topic_index]["status"] = "in_progress"
        
        # Проверяем завершение
        should_end = (
            next_action == "finish" or 
            new_topic_index >= len(updated_plan) or
            state["turn_id"] >= int(os.getenv("MAX_TURNS", "10"))
        )
        
        # Формируем internal_debate (формат: [agent]: thought\n)
        parts = []
        if state.get("router_thought"):
            parts.append(f"[Router]: {state['router_thought']}")
        if state.get("skeptic_thought"):
            parts.append(f"[Skeptic]: {state['skeptic_thought']}")
        if state.get("empath_thought"):
            parts.append(f"[Empath]: {state['empath_thought']}")
        parts.append(f"[Planner]: {internal_thought}")
        internal_debate = "\n".join(parts)
        
        # Обновляем behavioral context
        new_behavioral_context = state["behavioral_context"].copy()
        if result.new_protocol and result.new_protocol != "standard":
            new_behavioral_context["protocol"] = result.new_protocol
        
        if state.get("hallucination_detected"):
            new_behavioral_context["hallucination_count"] = new_behavioral_context.get("hallucination_count", 0) + 1
            
            # Если галлюцинация — модифицируем директиву для краткого спора/уточнения
            correct_answer = state.get("_skeptic_correct_answer", "")
            if correct_answer:
                directive = f"КРАТКО уточни/оспорь ложный факт: '{correct_answer}'. Не объясняй подробно, только уточни источник или факт. Затем задай следующий вопрос."
            else:
                directive = f"КРАТКО уточни ложный факт (кандидат выдумал информацию). Спроси откуда информация. Затем задай следующий вопрос."
            
            # Помечаем в internal_thought
            internal_thought = f"ГАЛЛЮЦИНАЦИЯ обнаружена! {internal_thought}"
        
        print(f"[Planner] action={next_action}, score={topic_score}, move_next={move_to_next}")
        if state.get("hallucination_detected"):
            print(f"[Planner] ГАЛЛЮЦИНАЦИЯ! Директива: {directive[:100]}")
        
        return {
            "interview_plan": updated_plan,
            "current_topic_index": new_topic_index,
            "planner_directive": directive,
            "planner_thought": internal_thought,
            "internal_debate": internal_debate,
            "behavioral_context": new_behavioral_context,
            "_move_to_next_topic": move_to_next,  # Флаг для Voice
            "should_end": should_end,
            "next_step": "end" if should_end else "respond"
        }
        
    except Exception as e:
        print(f"[Planner] Full plan error: {e}")
        return {
            "planner_directive": "Продолжай интервью",
            "planner_thought": f"Ошибка: {str(e)}",
            "internal_debate": f"[Planner]: Ошибка: {str(e)}",
            "next_step": "respond"
        }


def format_plan_status(plan: List[Dict]) -> str:
    """Форматирует план для промпта"""
    if not plan:
        return "План пуст"
    
    lines = []
    for i, topic in enumerate(plan):
        status_icon = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "skipped": "⏭️"
        }.get(topic["status"], "❓")
        
        score_str = f"({topic['score']}/10)" if topic.get("score") is not None else ""
        weak = topic.get("weak_answers", 0)
        weak_str = f" ⚠️{weak} слабых" if weak > 0 else ""
        lines.append(f"{i+1}. {status_icon} {topic['topic']} [{topic['difficulty']}] {score_str}{weak_str}")
    
    return "\n".join(lines)


def find_next_pending_topic(plan: List[Dict]) -> int:
    """Находит индекс следующей незавершенной темы"""
    for i, topic in enumerate(plan):
        if topic["status"] == "pending":
            return i
    return len(plan)

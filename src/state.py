"""
Interview State - единый источник истины для всей системы.
Это главный объект, который передается между узлами LangGraph.
"""

from typing import Annotated, List, Dict, Optional, Literal
from typing_extensions import TypedDict
import operator
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================
# Pydantic Models для Structured Output агентов
# Каждый агент генерирует internal_thought - краткую главную мысль
# ============================================================

class RouterOutput(BaseModel):
    """Structured output для Router агента"""
    intent: Literal["answer", "question", "off_topic", "stop"] = Field(
        description="Классифицированный интент сообщения кандидата"
    )
    internal_thought: str = Field(
        description="Краткая главная мысль: почему выбран этот интент (1-2 предложения)"
    )
    is_suspicious: bool = Field(
        default=False,
        description="Подозрительное техническое утверждение, требующее фактчекинга"
    )


class SkepticOutput(BaseModel):
    """Structured output для Skeptic агента (технический анализ)"""
    score: int = Field(ge=0, le=10, description="Техническая оценка ответа 0-10")
    accuracy: Literal["точный", "частично_верный", "неверный", "галлюцинация"] = Field(
        description="Точность технического ответа"
    )
    depth: Literal["поверхностный", "достаточный", "глубокий", "экспертный"] = Field(
        description="Глубина понимания темы"
    )
    internal_thought: str = Field(
        max_length=100,
        description="Главная мысль: 1 предложение о проблеме"
    )
    issues: Optional[List[str]] = Field(
        default=None,
        max_length=3,
        description="Макс 2-3 ключевые проблемы, каждая до 15 слов"
    )
    correct_answer: Optional[str] = Field(
        default=None,
        max_length=150,
        description="Краткий правильный ответ (если ошибся)"
    )
    contradiction_detected: bool = Field(
        default=False,
        description="True если противоречие с предыдущими словами"
    )
    fictional_term_detected: bool = Field(
        default=False,
        description="True если вымышленные термины/библиотеки"
    )


class EmpathOutput(BaseModel):
    """Structured output для Empath агента (поведенческий анализ)"""
    demeanor: Literal["normal", "verbose", "silent", "arrogant", "stuck", "nervous"] = Field(
        description="Манера общения кандидата"
    )
    clarity: int = Field(ge=1, le=10, description="Ясность изложения 1-10")
    honesty: int = Field(ge=1, le=10, description="Честность (признает ли незнание) 1-10")
    engagement: Literal["low", "medium", "high"] = Field(
        description="Уровень вовлеченности"
    )
    stress_level: Literal["low", "medium", "high"] = Field(
        description="Уровень стресса кандидата"
    )
    internal_thought: str = Field(
        description="Главная мысль: наблюдение о поведении/состоянии кандидата (1-2 предложения)"
    )
    recommended_protocol: Literal["standard", "rescue", "speedrun", "stress_test"] = Field(
        default="standard",
        description="Рекомендуемый протокол ведения интервью"
    )


class PlannerOutput(BaseModel):
    """Structured output для Planner агента (стратегическое планирование)"""
    topic_score: Optional[int] = Field(
        default=None, ge=0, le=10,
        description="Оценка по текущей теме или None"
    )
    next_action: Literal["continue", "next_topic", "answer_question", "handle_offtopic", "handle_hallucination", "finish"] = Field(
        description="Следующее действие"
    )
    difficulty_change: Literal["increase", "decrease", "keep"] = Field(
        default="keep",
        description="Изменение сложности вопросов"
    )
    new_protocol: Literal["standard", "rescue", "speedrun", "stress_test"] = Field(
        default="standard",
        description="Новый протокол если нужно изменить"
    )
    directive: str = Field(
        description="Четкая инструкция для Voice что делать/сказать"
    )
    internal_thought: str = Field(
        description="Главная мысль: стратегическое решение и его причина (1-2 предложения)"
    )


class VoiceOutput(BaseModel):
    """Structured output для Voice агента (генерация ответа)"""
    message: str = Field(
        description="Сообщение интервьюера для кандидата"
    )
    internal_thought: str = Field(
        description="Главная мысль: почему выбран такой тон/содержание ответа (1-2 предложения)"
    )


class ReporterOutput(BaseModel):
    """Structured output для Reporter агента (финальный отчет)"""
    assessed_grade: Literal["Junior", "Middle", "Senior"] = Field(
        description="Реальный оценённый уровень кандидата"
    )
    hiring_recommendation: Literal["Strong Hire", "Hire", "No Hire"] = Field(
        description="Рекомендация по найму"
    )
    confidence_score: int = Field(ge=0, le=100, description="Уверенность в оценке 0-100%")
    verdict_reasoning: str = Field(
        description="Обоснование вердикта (2-3 предложения)"
    )
    clarity_score: int = Field(ge=1, le=10, description="Оценка ясности изложения")
    honesty_score: int = Field(ge=1, le=10, description="Оценка честности")
    engagement_score: int = Field(ge=1, le=10, description="Оценка вовлеченности")
    soft_skills_notes: str = Field(description="Заметки о soft skills")
    roadmap: List[str] = Field(description="Рекомендации по развитию (темы)")
    resources: List[str] = Field(description="Полезные ресурсы для изучения")
    internal_thought: str = Field(
        description="Главная мысль: ключевой вывод по кандидату (1-2 предложения)"
    )


# ============================================================
# Enums и TypedDict для состояния
# ============================================================

class TopicStatus(str, Enum):
    """Статус темы в плане интервью"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"

class TopicDifficulty(str, Enum):
    """Уровень сложности темы"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class Demeanor(str, Enum):
    """Манера общения"""
    NORMAL = "normal"
    VERBOSE = "verbose"
    SILENT = "silent"
    ARROGANT = "arrogant"
    STUCK = "stuck"

class Protocol(str, Enum):
    """Протокол поведения"""
    STANDARD = "standard"
    RESCUE = "rescue"
    SPEEDRUN = "speedrun"
    STRESS_TEST = "stress_test"

class StressLevel(str, Enum):
    """Уровень стресса"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class InterviewTopic(TypedDict):
    """Тема в плане интервью"""
    id: int
    topic: str
    difficulty: TopicDifficulty  # TopicDifficulty
    status: TopicStatus  # TopicStatus
    score: Optional[int]  # 0-10
    feedback: str
    correct_answer: Optional[str]  # Правильный ответ (для Knowledge Gaps)


class CandidateMetadata(TypedDict):
    """Данные о кандидате"""
    name: str
    role: str
    target_grade: str  # Junior / Middle / Senior
    experience: str


class BehavioralContext(TypedDict):
    """Психологический контекст кандидата"""
    demeanor: Demeanor  # normal, verbose, silent, arrogant, stuck
    protocol: Protocol  # standard, rescue, speedrun, stress_test
    stress_level: StressLevel  # low, medium, high
    hallucination_count: int
    off_topic_count: int


class TurnLog(TypedDict):
    """Запись одного хода для JSON-лога"""
    turn_id: int
    agent_visible_message: str
    user_message: str
    internal_thoughts: str


class SkillAssessment(TypedDict):
    """Оценка навыка"""
    topic: str
    score: int
    confirmed: bool
    feedback: str
    correct_answer: Optional[str]


class SoftSkillsAssessment(TypedDict):
    """Оценка soft skills"""
    clarity: int  # 1-10
    honesty: int  # 1-10
    engagement: int  # 1-10
    notes: str


class FinalFeedback(TypedDict):
    """Финальный фидбэк"""
    # Вердикт
    assessed_grade: str  # Junior / Middle / Senior
    hiring_recommendation: str  # Hire / No Hire / Strong Hire
    confidence_score: int  # 0-100
    
    # Hard Skills
    confirmed_skills: List[SkillAssessment]
    knowledge_gaps: List[SkillAssessment]
    
    # Soft Skills
    soft_skills: SoftSkillsAssessment
    
    # Roadmap
    roadmap: List[str]
    resources: List[str]


class Message(TypedDict):
    """Сообщение в истории"""
    role: str  # user, assistant, system
    content: str

class UserIntent(str, Enum):
    """Интент пользователя"""
    ANSWER = "answer"  # Ответ на вопрос
    QUESTION = "question"  # Встречный вопрос
    OFF_TOPIC = "off_topic"  # Оффтоп
    STOP = "stop"  # Завершение интервью

class StateStep(str, Enum):
    """Шаг состояния"""
    ROUTER = "router"
    ANALYZE = "analyze"
    PLAN = "plan"
    RESPOND = "respond"
    END = "end"

class InterviewState(TypedDict):
    """
    Главное состояние системы.
    Передается между всеми узлами LangGraph.
    """
    # === Данные кандидата ===
    metadata: CandidateMetadata
    
    # === План интервью (динамический To-Do List) ===
    interview_plan: List[InterviewTopic]
    current_topic_index: int
    
    # === Психологический контекст ===
    behavioral_context: BehavioralContext
    
    # === История и логи ===
    messages: Annotated[List[Message], operator.add]  # История сообщений (аккумулируется)
    internal_debate: str  # Агрегированные мысли всех агентов на текущем ходе
    turns_log: List[TurnLog]  # Финальный JSON-лог
    turn_id: int
    
    # === Анализ текущего хода ===
    current_user_message: str
    user_intent: UserIntent  # UserIntent
    skeptic_analysis: str
    empath_analysis: str
    planner_directive: str
    
    # === Internal Thoughts от каждого агента (structured output) ===
    router_thought: str  # Мысль Router: почему выбран интент
    skeptic_thought: str  # Мысль Skeptic: техническая оценка
    empath_thought: str  # Мысль Empath: поведенческая оценка
    planner_thought: str  # Мысль Planner: стратегическое решение
    voice_thought: str  # Мысль Voice: выбор тона ответа
    
    # === Служебные флаги ===
    next_step: StateStep  # router, analyze, plan, respond, end
    should_end: bool
    hallucination_detected: bool
    
    # === Текущий ответ Voice ===
    current_response: str  # Последний ответ Voice для возврата
    
    # === Финальный фидбэк ===
    final_feedback: Optional[FinalFeedback]
    final_report_string: str  # Отформатированный отчёт Reporter


def create_initial_state(
    name: str,
    role: str,
    grade: str,
    experience: str
) -> InterviewState:
    """Создает начальное состояние для нового интервью"""
    return InterviewState(
        metadata=CandidateMetadata(
            name=name,
            role=role,
            target_grade=grade,
            experience=experience
        ),
        interview_plan=[],
        current_topic_index=0,
        behavioral_context=BehavioralContext(
            demeanor="normal",
            protocol="standard",
            stress_level="low",
            hallucination_count=0,
            off_topic_count=0
        ),
        messages=[],
        internal_debate="",
        turns_log=[],
        turn_id=0,
        current_user_message="",
        user_intent="answer",
        skeptic_analysis="",
        empath_analysis="",
        planner_directive="",
        # Internal thoughts от каждого агента
        router_thought="",
        skeptic_thought="",
        empath_thought="",
        planner_thought="",
        voice_thought="",
        # Служебные флаги
        next_step="router",
        should_end=False,
        hallucination_detected=False,
        # Ответ Voice и Reporter
        current_response="",
        final_feedback=None,
        final_report_string=""
    )

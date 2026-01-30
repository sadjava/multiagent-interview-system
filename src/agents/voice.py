"""
Voice (Interviewer) - агент, который ведет диалог с кандидатом.
Формирует сообщения, видимые пользователю.
Ведёт себя как профессиональный интервьюер — НЕ учитель!
"""

import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..state import InterviewState, VoiceOutput


VOICE_PROMPT = """Ты - Интервьюер на техническом собеседовании.
Твоя роль — ОЦЕНИВАТЬ кандидата, а НЕ ОБУЧАТЬ его.

🚨 КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. НИКОГДА не объясняй правильный ответ кандидату
2. НИКОГДА не исправляй ошибки кандидата подробно
3. Ты проводишь СОБЕСЕДОВАНИЕ, а не УРОК

📝 ПРАВИЛА ВОПРОСОВ:
- Задавай ОДИН простой вопрос за раз
- Максимум 1-2 вопроса в сообщении
- НЕ задавай 3-5 вопросов сразу — это неудобно и нечитаемо
- Формат: вопрос → ответ кандидата → следующий вопрос
- Можно: "Расскажите про X. После этого покажите пример кода."
- Нельзя: "Расскажите про X, Y, Z, приведите примеры A, B, C и объясните D."

ТЕКУЩИЙ ХОД: {turn_id}
{first_turn_instruction}

ДАННЫЕ О КАНДИДАТЕ:
Имя: {name}
Позиция: {role}
Грейд: {grade}

ДИРЕКТИВА ОТ СТРАТЕГА:
{directive}

ПРОТОКОЛ: {protocol}
- standard: обычный режим
- rescue: кандидат застрял, упрости вопрос
- speedrun: блиц-вопросы
- stress_test: сложные вопросы

ТЕКУЩАЯ ТЕМА: {current_topic}
СЛОЖНОСТЬ: {difficulty}

ИНТЕНТ: {user_intent}
СООБЩЕНИЕ КАНДИДАТА: {user_message}

ИСТОРИЯ (последние сообщения):
{history}

{special_instructions}

ЗАДАЧА:
Сгенерируй ОДНО сообщение для кандидата.

Правила по интенту:
- answer → кратко отреагируй (1 предложение) + задай ОДИН следующий вопрос
- question → ответь КРАТКО (1-2 предложения) + вернись к текущей теме
- off_topic → вежливо верни к теме (без обсуждения оффтопа)

Ответ:
- message: сообщение для кандидата (короткое, 1-2 вопроса максимум)
- internal_thought: почему выбран такой подход
"""


def get_voice_llm():
    """Возвращает LLM для Voice с structured output"""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
        temperature=0.7,
        reasoning_effort="low",
        max_completion_tokens=2000
    ).with_structured_output(VoiceOutput)


def voice_node(state: InterviewState) -> Dict[str, Any]:
    """
    Узел Voice - генерирует ответ интервьюера.
    Используется для ВСЕХ ходов, включая первый.
    """
    user_intent = state.get("user_intent", "answer")
    user_message = state.get("current_user_message", "")
    turn_id = state["turn_id"]
    
    # Получаем тему
    current_topic = "Общие вопросы"
    difficulty = "medium"
    if state["interview_plan"] and state["current_topic_index"] < len(state["interview_plan"]):
        topic = state["interview_plan"][state["current_topic_index"]]
        current_topic = topic["topic"]
        difficulty = topic["difficulty"]
    
    # История
    history = format_history(state["messages"], last_n=6)
    
    # Специальные инструкции
    special_instructions = get_special_instructions(state)
    
    # Инструкция для первого хода
    first_turn_instruction = ""
    if turn_id == 0:
        first_turn_instruction = """
⭐ ЭТО ПЕРВЫЙ ХОД — нужно поприветствовать кандидата!
- Коротко поздоровайся (1 предложение)
- Упомяни что-то из опыта кандидата
- Задай ОДИН первый вопрос по теме
"""
    
    prompt = ChatPromptTemplate.from_template(VOICE_PROMPT)
    llm = get_voice_llm()
    chain = prompt | llm
    
    try:
        result: VoiceOutput = chain.invoke({
            "turn_id": turn_id,
            "first_turn_instruction": first_turn_instruction,
            "name": state["metadata"]["name"],
            "role": state["metadata"]["role"],
            "grade": state["metadata"]["target_grade"],
            "directive": state.get("planner_directive", "Продолжай интервью"),
            "protocol": state["behavioral_context"]["protocol"],
            "current_topic": current_topic,
            "difficulty": difficulty,
            "user_intent": user_intent,
            "user_message": user_message or "(первый ход, кандидат ещё не отвечал)",
            "history": history,
            "special_instructions": special_instructions
        })
        
        response = result.message
        voice_thought = result.internal_thought
        
        print(f"[Voice] Ход {turn_id}: ответ сгенерирован ({len(response)} символов)")
        print(f"[Voice] Мысль: {voice_thought}")
        
        # Собираем internal_thoughts (формат: [agent]: thought\n)
        internal_debate = state.get("internal_debate", "")
        if internal_debate:
            internal_thoughts = f"{internal_debate}\n[Voice]: {voice_thought}"
        else:
            internal_thoughts = f"[Voice]: {voice_thought}"
        
        # Обновляем историю
        new_messages = []
        if user_message:
            new_messages.append({"role": "user", "content": user_message})
        new_messages.append({"role": "assistant", "content": response})
        
        return {
            "messages": new_messages,
            "voice_thought": voice_thought,
            "current_response": response,
            "internal_debate": internal_thoughts,
            "turn_id": turn_id + 1,
            "next_step": "router"
        }
        
    except Exception as e:
        print(f"[Voice] Ошибка: {e}")
        error_response = "Хорошо, давайте перейдём к следующему вопросу."
        error_thought = f"Ошибка: {str(e)}"
        return {
            "messages": [{"role": "assistant", "content": error_response}],
            "voice_thought": error_thought,
            "current_response": error_response,
            "internal_debate": f"[Voice]: {error_thought}",
            "turn_id": turn_id + 1,
            "next_step": "router"
        }


def format_history(messages: List[Dict], last_n: int = 6) -> str:
    """Форматирует историю сообщений"""
    if not messages:
        return "Начало интервью"
    
    recent = messages[-last_n:]
    lines = []
    for msg in recent:
        role = "Кандидат" if msg["role"] == "user" else "Интервьюер"
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        lines.append(f"{role}: {content}")
    
    return "\n".join(lines)


def get_special_instructions(state: InterviewState) -> str:
    """Возвращает специальные инструкции"""
    instructions = []
    
    # Галлюцинация - НЕ объяснять
    if state.get("hallucination_detected"):
        instructions.append(
            "⚠️ ОБНАРУЖЕНА ОШИБКА/ГАЛЛЮЦИНАЦИЯ. "
            "НЕ объясняй правильный ответ! Варианты:\n"
            "- Уточняющий вопрос: 'Интересно, а можете привести пример?'\n"
            "- Переход к другой теме: 'Хорошо, давайте к следующему вопросу.'"
        )
    
    # Оффтоп
    if state.get("user_intent") == "off_topic":
        instructions.append(
            "Кандидат ушёл от темы. Вежливо верни к интервью."
        )
    
    # Встречный вопрос
    if state.get("user_intent") == "question":
        instructions.append(
            "Кандидат задал вопрос. Ответь КРАТКО (1-2 предложения) и продолжи текущую тему."
        )
    
    # Переход к следующей теме (от Planner)
    if state.get("_move_to_next_topic"):
        instructions.append(
            "📌 ПЕРЕХОДИМ К СЛЕДУЮЩЕЙ ТЕМЕ. Плавно заверши текущую и начни новую."
        )
    
    # Протоколы
    if state["behavioral_context"]["protocol"] == "rescue":
        instructions.append("Кандидат застрял. Упрости вопрос или дай подсказку.")
    
    if state["behavioral_context"]["protocol"] == "speedrun":
        instructions.append("Блиц-режим. Короткие вопросы.")
    
    return "\n".join(instructions) if instructions else ""

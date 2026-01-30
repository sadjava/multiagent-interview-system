"""
Input Router - классификатор интента пользователя.
Использует LLM с strict structured output для надёжной классификации.
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..state import InterviewState, RouterOutput


ROUTER_PROMPT = """Классифицируй сообщение кандидата на интервью.

ВОПРОС ИНТЕРВЬЮЕРА: {question}

СООБЩЕНИЕ КАНДИДАТА: {message}

ИНТЕНТЫ (выбери ОДИН):
- answer: кандидат ОТВЕЧАЕТ на вопрос (любой ответ, даже неполный или неверный)
- question: кандидат ЗАДАЁТ встречный вопрос интервьюеру
- off_topic: кандидат говорит НЕ по теме, уходит в сторону
- stop: кандидат хочет ЗАВЕРШИТЬ интервью (стоп, хватит, достаточно, закончим)

ПРАВИЛА:
- Если кандидат пытается ответить на вопрос → answer
- Если кандидат спрашивает "а что такое X?" → question
- Если кандидат говорит о чём-то своём, не связанном с вопросом → off_topic
- Слова "стоп", "достаточно", "хватит", "закончим" → stop

Выбери intent и напиши краткую мысль (1 предложение).
"""


def get_router_llm():
    """LLM для классификации с strict structured output."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
        temperature=0,
        reasoning_effort="minimal",
        max_completion_tokens=1000
    ).with_structured_output(RouterOutput)


def router_node(state: InterviewState) -> Dict[str, Any]:
    """
    Узел роутера - классифицирует интент через LLM.
    
    Интенты:
    - answer: технический ответ
    - question: встречный вопрос
    - off_topic: оффтоп
    - stop: завершение
    """
    user_message = state.get("current_user_message", "")
    turn_id = state.get("turn_id", 0)
    
    # Первый ход — кандидат ещё не отвечал
    if turn_id == 0 or not user_message:
        print(f"[Router] Первый ход (turn_id={turn_id}) — пропуск классификации")
        return {
            "user_intent": "answer",
            "next_step": "quick_respond",
            "router_thought": "Первый ход — генерируем приветствие и первый вопрос."
        }
    
    # Последний вопрос интервьюера (полный текст)
    last_question = ""
    for msg in reversed(state.get("messages", [])):
        if msg["role"] == "assistant":
            last_question = msg["content"]
            break
    
    # LLM классификация
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    llm = get_router_llm()
    chain = prompt | llm
    
    try:
        result: RouterOutput = chain.invoke({
            "question": last_question or "Начало интервью",
            "message": user_message
        })
        
        intent = result.intent
        internal_thought = result.internal_thought
        
        print(f"[Router] Интент: {intent}")
        print(f"[Router] Мысль: {internal_thought}")
        
        # Определяем следующий шаг
        if intent == "answer":
            next_step = "analyze"
        elif intent == "stop":
            next_step = "end"
        else:
            next_step = "quick_respond"
        
        return {
            "user_intent": intent,
            "next_step": next_step,
            "router_thought": internal_thought,
        }
        
    except Exception as e:
        print(f"[Router] Ошибка: {e}, fallback → answer")
        return {
            "user_intent": "answer",
            "next_step": "analyze",
            "router_thought": f"Ошибка классификации: {str(e)[:50]}, fallback на answer"
        }


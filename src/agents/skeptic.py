"""
Technical Skeptic - агент для оценки Hard Skills.
"ПЛОХОЙ КОП" / Техлид - ЖЁСТКО проверяет техническую корректность.
Ищет ошибки, противоречия, вымышленные термины.
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..state import InterviewState, SkepticOutput


SKEPTIC_PROMPT = """Оцени ответ кандидата. ВНИМАТЕЛЬНО проверь ВСЕ термины и технологии!

ВОПРОС: {question}
ОТВЕТ: {answer}

ШКАЛА: 8-10 отлично, 6-7 хорошо, 4-5 частично, 1-3 слабо, 0 провал

КРИТЕРИИ:
- Отвечает ли на ЗАДАННЫЙ вопрос?
- Есть ли конкретика?
- Есть ли ГРУБЫЕ ошибки?

🔍 ОБЯЗАТЕЛЬНО ПРОВЕРЬ ВСЕ ТЕРМИНЫ И ТЕХНОЛОГИИ:
1. ВЕРСИИ: Python 4.0? Django 5.x? Проверь — существуют ли они?
2. БИБЛИОТЕКИ: quantum-arch? neo-memoization? Проверь — есть ли такие библиотеки?
3. PEP/RFC: PEP 9999? RFC 8888? Проверь — существуют ли они?
4. ТЕРМИНЫ: Незнакомые термины? Проверь — это реальная технология или выдумка?
5. ФУНКЦИИ/МЕТОДЫ: Несуществующие методы API? Проверь документацию!

🚨 КРИТИЧЕСКИ ВАЖНО - ЛОЖНЫЕ ФАКТЫ (ГАЛЛЮЦИНАЦИИ):
- ЛОЖНЫЕ ФАКТЫ — это КРИТИЧЕСКОЕ нарушение! Реагируй ЖЁСТКО!
- Если кандидат упоминает что-то, что НЕ СУЩЕСТВУЕТ — это галлюцинация!
- НЕ СУЩЕСТВУЮЩИЕ ТЕХНОЛОГИИ = галлюцинация!
- ОБЯЗАТЕЛЬНО помечай как "галлюцинация" и указывай правильный факт в correct_answer
- Галлюцинация = score 0-1 (максимально жёсткая оценка!)
- fictional_term_detected = True если упомянута несуществующая технология/библиотека

ПРИМЕРЫ ГАЛЛЮЦИНАЦИЙ (НЕСУЩЕСТВУЮЩИЕ ТЕХНОЛОГИИ):
- "Python 4.0 вышел в 2024" → ГАЛЛЮЦИНАЦИЯ (Python 4.0 не существует, последняя версия Python 3.x)
- "Django 5.0 поддерживает..." → ГАЛЛЮЦИНАЦИЯ (если версии нет, проверь актуальную версию)
- "Библиотека quantum-arch делает..." → ГАЛЛЮЦИНАЦИЯ (если библиотеки нет в PyPI/официальных источниках)
- "В PEP 9999 написано..." → ГАЛЛЮЦИНАЦИЯ (если PEP не существует)
- "Использую neo-memoization для кэширования" → ГАЛЛЮЦИНАЦИЯ (если библиотеки нет)
- "Фреймворк FlaskORM для ORM" → ГАЛЛЮЦИНАЦИЯ (если фреймворка нет)

ВАЖНО: Критика должна быть ОБОСНОВАННОЙ и СВЯЗАННОЙ с вопросом!
Не придумывай проблемы. Если ответ хороший — так и напиши.
НО: Если видишь подозрительный термин — ОБЯЗАТЕЛЬНО проверь его существование!

ОТВЕТ:
- score (0-10): галлюцинация = 0-1 (максимально жёстко!)
- accuracy: точный/частично_верный/неверный/галлюцинация
- depth: поверхностный/достаточный/глубокий/экспертный
- internal_thought: краткая оценка + проблемы (если есть) в 1-2 предложениях. Если галлюцинация — укажи что именно выдумано!
- issues: обязательно укажи если галлюцинация (что именно выдумано: версия/библиотека/термин)
- correct_answer: ОБЯЗАТЕЛЬНО укажи правильный факт если галлюцинация (например: "Python 4.0 не существует, последняя версия Python 3.12")
- contradiction_detected: bool
- fictional_term_detected: bool (ОБЯЗАТЕЛЬНО True если упомянута несуществующая технология/библиотека/версия)
"""


def get_skeptic_llm():
    """LLM для анализа со structured output."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
        temperature=0.1,
        reasoning_effort="medium",
        max_completion_tokens=10000  # Достаточно для reasoning + output
    ).with_structured_output(SkepticOutput)


def skeptic_node(state: InterviewState) -> Dict[str, Any]:
    """
    Узел Скептика - ЖЁСТКО анализирует техническую корректность.
    Ищет: ошибки, галлюцинации, противоречия, вымышленные термины.
    """
    user_message = state["current_user_message"]
    user_intent = state.get("user_intent", "answer")

    if user_intent not in ["answer"]:
        print(f"[Skeptic] Пропуск: интент '{user_intent}'")
        return {
            "skeptic_analysis": f"[Skeptic]: Пропуск (интент: {user_intent})",
            "skeptic_thought": f"Интент '{user_intent}' — не технический ответ."
        }

    # Текущая тема
    topic_name = "Общие вопросы"
    difficulty = "medium"
    if state["interview_plan"] and state["current_topic_index"] < len(state["interview_plan"]):
        topic = state["interview_plan"][state["current_topic_index"]]
        topic_name = topic["topic"]
        difficulty = topic["difficulty"]

    # Последний вопрос интервьюера (полный текст)
    last_question = "Начало интервью"
    for msg in reversed(state["messages"]):
        if msg["role"] == "assistant":
            last_question = msg["content"]
            break

    # === LLM Анализ ===
    prompt = ChatPromptTemplate.from_template(SKEPTIC_PROMPT)
    llm = get_skeptic_llm()
    chain = prompt | llm

    try:
        result: SkepticOutput = chain.invoke({
            "role": state["metadata"]["role"],
            "grade": state["metadata"]["target_grade"],
            "topic": topic_name,
            "question": last_question,
            "answer": user_message
        })

        # Формируем полную мысль с проблемами для логов
        issues_list = result.issues or []
        
        # Собираем всё в одну строку для internal_thought
        thought_parts = [result.internal_thought]
        if issues_list:
            thought_parts.append(f"Проблемы: {'; '.join(issues_list[:3])}")
        if result.accuracy == "галлюцинация":
            thought_parts.append("ГАЛЛЮЦИНАЦИЯ!")
        if getattr(result, 'contradiction_detected', False):
            thought_parts.append("ПРОТИВОРЕЧИЕ!")
        if getattr(result, 'fictional_term_detected', False):
            thought_parts.append("ВЫМЫШЛЕННЫЙ ТЕРМИН!")
        
        full_thought = f"[{result.score}/10] " + " | ".join(thought_parts)

        hallucination_detected = (
            result.accuracy == "галлюцинация" or 
            getattr(result, 'fictional_term_detected', False)
        )
        
        contradiction_detected = getattr(result, 'contradiction_detected', False)

        print(f"[Skeptic] {full_thought}")

        return {
            "skeptic_analysis": f"[Skeptic]: {full_thought}",
            "skeptic_thought": full_thought,
            "hallucination_detected": hallucination_detected,
            "_skeptic_score": result.score,
            "_skeptic_accuracy": result.accuracy,
            "_skeptic_depth": result.depth,
            "_skeptic_correct_answer": result.correct_answer,
            "_skeptic_issues": issues_list,
            "_contradiction_detected": contradiction_detected,
            "_fictional_term_detected": getattr(result, 'fictional_term_detected', False)
        }

    except Exception as e:
        error_thought = f"Ошибка анализа: {str(e)}"
        print(f"[Skeptic] Ошибка: {e}")

        return {
            "skeptic_analysis": f"[Skeptic]: {error_thought}",
            "skeptic_thought": error_thought,
            "hallucination_detected": False
        }

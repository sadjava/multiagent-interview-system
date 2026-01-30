"""
Behavioral Empath - агент для оценки Soft Skills.
"ХОРОШИЙ КОП" / HR - ЗАЩИЩАЕТ кандидата, ищет позитив.
Находит оправдания, поддерживает, видит потенциал.
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..state import InterviewState, EmpathOutput


EMPATH_PROMPT = """Оцени SOFT SKILLS кандидата (НЕ техническую часть!).

СООБЩЕНИЕ КАНДИДАТА: {message}

ОЦЕНИВАЙ ТОЛЬКО:
1. КАК говорит (не ЧТО говорит технически)
2. Манера общения, уверенность, стресс
3. Структурированность речи
4. Вовлечённость в диалог

🔍 ОБЯЗАТЕЛЬНО ПРОВЕРЬ ЧЕСТНОСТЬ:

1. ИСПОЛЬЗОВАНИЕ AI:
   - Упоминает ли ChatGPT, GPT, AI-ассистентов, Copilot?
   - Говорит ли "AI подсказал", "спросил у GPT", "использовал Copilot"?
   - Если ДА → это НЕЧЕСТНОСТЬ! honesty = 1-3 (очень низкая!)
   - Кандидат должен отвечать САМ, не используя AI!

2. ЛИЧНЫЙ ОПЫТ:
   - Есть ли конкретные примеры из СВОЕЙ практики?
   - Говорит ли "я делал", "в моём проекте", "я использовал"?
   - Или только общие фразы без личного опыта?
   - Если только общие фразы без примеров → возможно, не сам писал код

3. ПРИЗНАНИЕ НЕЗНАНИЯ:
   - Признаёт ли если не знает? ("не уверен", "не помню", "не знаю")
   - Или выдумывает и выкручивается?
   - Признание незнания = честность (honesty повышается)
   - Выдумки = нечестность (honesty снижается)

ПРИМЕРЫ НЕЧЕСТНОСТИ:
- "ChatGPT подсказал мне..." → НЕЧЕСТНОСТЬ! honesty = 1-2
- "Использовал Copilot для этого" → НЕЧЕСТНОСТЬ! honesty = 1-2
- "AI сказал что..." → НЕЧЕСТНОСТЬ! honesty = 1-2
- Только общие фразы без личного опыта → подозрительно, honesty = 3-5

НЕ ОЦЕНИВАЙ техническую правильность — это делает Skeptic!

ОТВЕТ:
- demeanor: normal/verbose/silent/arrogant/stuck/nervous
- clarity (1-10): насколько понятно излагает мысли
- honesty (1-10): 
  * 1-3: использует AI, выдумывает, нечестен
  * 4-6: подозрительно, мало личного опыта
  * 7-10: честен, признаёт незнание, есть личный опыт
- engagement: low/medium/high — активность в диалоге
- stress_level: low/medium/high
- internal_thought: 1 предложение о ПОВЕДЕНИИ + честности (если есть проблемы с AI/опытом — укажи!)
- recommended_protocol: standard/rescue/speedrun/stress_test
"""


def get_empath_llm():
    """Возвращает LLM для Эмпата с structured output"""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini"),
        temperature=0.3,
        reasoning_effort="low",
        max_completion_tokens=2000
    ).with_structured_output(EmpathOutput)


def empath_node(state: InterviewState) -> Dict[str, Any]:
    """
    Узел Эмпата - оценивает SOFT SKILLS (поведение, манера общения).
    НЕ оценивает техническую правильность.
    """
    user_message = state["current_user_message"]
    
    prompt = ChatPromptTemplate.from_template(EMPATH_PROMPT)
    llm = get_empath_llm()
    chain = prompt | llm
    
    try:
        result: EmpathOutput = chain.invoke({
            "message": user_message
        })
        
        demeanor = result.demeanor
        clarity = result.clarity
        honesty = result.honesty
        engagement = result.engagement
        stress_level = result.stress_level
        internal_thought = result.internal_thought
        recommended_protocol = result.recommended_protocol
        
        empath_analysis = (
            f"[Empath]: {internal_thought} "
            f"(Ясность: {clarity}/10, Честность: {honesty}/10, Вовлечённость: {engagement})"
        )
        
        # Обновляем behavioral_context
        new_behavioral_context = state["behavioral_context"].copy()
        new_behavioral_context["demeanor"] = demeanor
        new_behavioral_context["stress_level"] = stress_level
        
        if recommended_protocol != "standard":
            new_behavioral_context["protocol"] = recommended_protocol
        
        print(f"[Empath] demeanor={demeanor}, clarity={clarity}, engagement={engagement}")
        print(f"[Empath] Мысль: {internal_thought}")
        
        return {
            "empath_analysis": empath_analysis,
            "empath_thought": internal_thought,
            "behavioral_context": new_behavioral_context,
            "_empath_clarity": clarity,
            "_empath_honesty": honesty,
            "_empath_engagement": engagement
        }
        
    except Exception as e:
        print(f"[Empath] Ошибка: {e}")
        error_thought = f"Ошибка анализа: {str(e)}"
        return {
            "empath_analysis": f"[Empath]: {error_thought}",
            "empath_thought": error_thought
        }

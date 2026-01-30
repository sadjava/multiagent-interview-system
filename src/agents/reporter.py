"""
Final Reporter - агент для генерации финального фидбэка.
Создает структурированный отчет по результатам интервью.
Анализирует ВСЮ историю диалога и internal_thoughts агентов.
"""

import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..state import InterviewState, ReporterOutput, FinalFeedback, SkillAssessment, SoftSkillsAssessment


REPORTER_PROMPT = """Ты - Final Reporter. Создай ОБЪЕКТИВНЫЙ отчет по интервью.

ДАННЫЕ О КАНДИДАТЕ:
Имя: {name}
Позиция: {role}
Заявленный грейд: {grade}
Опыт: {experience}

================================================================================
ПОЛНАЯ ИСТОРИЯ ИНТЕРВЬЮ (ВАЖНО! Внимательно изучи каждый ход):
================================================================================
{full_dialogue}

================================================================================
АНАЛИЗ АГЕНТОВ ПО ХОДАМ (internal_thoughts):
================================================================================
{agents_analysis}

================================================================================
СТАТИСТИКА ИНТЕРВЬЮ:
================================================================================
- Всего вопросов задано: {total_questions}
- Вопросов БЕЗ ответа (кандидат ушёл от темы): {unanswered_questions}
- Галлюцинаций: {hallucination_count}
- Оффтопов: {off_topic_count}
- Противоречий: {contradiction_count}

КРИТИЧЕСКИЕ ПРОБЛЕМЫ:
{critical_issues}

ОЦЕНКИ ПО ТЕМАМ:
{topics_summary}

================================================================================
ПРАВИЛА ОЦЕНКИ (СТРОГО СОБЛЮДАЙ!):
================================================================================

HARD SKILLS:
- Если кандидат НЕ ОТВЕТИЛ на вопросы → оценка 1-3 по теме!
- Если ответ был off-topic → это НЕ ответ, оценка 1-3!
- Галлюцинации = автоматически "No Hire"
- Грубые технические ошибки = "No Hire"

SOFT SKILLS:
- clarity (ясность): Если НЕ отвечал на вопросы, уходил от темы → 1-4!
- honesty (честность): Галлюцинации = нечестность → 1-3! Противоречия = снижение!
- engagement (вовлечённость): Если только рассказывал свои истории, не отвечая → 1-4!

HIRING DECISION:
- Если кандидат НЕ ОТВЕТИЛ ни на один вопрос → "No Hire" (даже с большим опытом!)
- Если большинство ответов были off-topic → "No Hire"
- Опыт в резюме НЕ заменяет ответы на интервью!

CONFIDENCE_SCORE (основывается на ДАННЫХ, а не опыте):
- Если 3+ содержательных ответа → 80-95%
- Если 1-2 содержательных ответа → 60-80%
- Если 0 содержательных ответов → оценка на основе "нет данных", уверенность 30-50%

================================================================================
ТВОЯ ЗАДАЧА:
================================================================================
Изучи ПОЛНУЮ историю выше. Ответь на вопросы:
1. Сколько раз кандидат РЕАЛЬНО ответил на заданный вопрос?
2. Что говорил Skeptic о качестве ответов?
3. Были ли уходы от темы?

На основе ЭТОГО (а не резюме!) сгенерируй отчет.

Сгенерируй:
- assessed_grade: реальный уровень (на основе ОТВЕТОВ, не резюме!)
- hiring_recommendation: "Strong Hire" / "Hire" / "No Hire"
- confidence_score: 0-100%
- verdict_reasoning: 2-3 предложения (упомяни конкретно: ответил ли на вопросы!)
- clarity_score: 1-10
- honesty_score: 1-10
- engagement_score: 1-10
- soft_skills_notes: описание поведения
- roadmap: что изучить (3+)
- resources: ссылки
- internal_thought: ключевой вывод
"""


def get_reporter_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.2,
        reasoning_effort="low",
        max_completion_tokens=4000  # Отчёт подробный
    ).with_structured_output(ReporterOutput)


def reporter_node(state: InterviewState) -> Dict[str, Any]:
    """Генерирует финальный отчет на основе ПОЛНОЙ истории интервью."""
    
    # Собираем полный контекст
    full_dialogue = format_full_dialogue(state["turns_log"])
    agents_analysis = format_agents_analysis(state["turns_log"])
    critical_issues = collect_critical_issues(state)
    topics_summary = format_topics_summary(state["interview_plan"])
    
    # Подсчёт статистики
    total_questions = len(state["turns_log"])
    unanswered = count_unanswered_questions(state["turns_log"])
    
    prompt = ChatPromptTemplate.from_template(REPORTER_PROMPT)
    llm = get_reporter_llm()
    chain = prompt | llm
    
    try:
        result: ReporterOutput = chain.invoke({
            "name": state["metadata"]["name"],
            "role": state["metadata"]["role"],
            "grade": state["metadata"]["target_grade"],
            "experience": state["metadata"]["experience"],
            "full_dialogue": full_dialogue,
            "agents_analysis": agents_analysis,
            "total_questions": total_questions,
            "unanswered_questions": unanswered,
            "hallucination_count": state["behavioral_context"].get("hallucination_count", 0),
            "off_topic_count": state["behavioral_context"].get("off_topic_count", 0),
            "contradiction_count": state["behavioral_context"].get("contradiction_count", 0),
            "critical_issues": critical_issues,
            "topics_summary": topics_summary
        })
        
        print(f"[Reporter] Отчет: {result.hiring_recommendation}, уверенность: {result.confidence_score}%")
        
        # Собираем skills
        confirmed_skills = []
        knowledge_gaps = []
        
        for topic in state["interview_plan"]:
            score = topic.get("score")
            if score is not None:
                skill = SkillAssessment(
                    topic=topic["topic"],
                    score=score,
                    confirmed=score >= 7,
                    feedback=topic.get("feedback", ""),
                    correct_answer=topic.get("correct_answer")
                )
                if score >= 7:
                    confirmed_skills.append(skill)
                else:
                    knowledge_gaps.append(skill)
        
        feedback: FinalFeedback = {
            "assessed_grade": result.assessed_grade,
            "hiring_recommendation": result.hiring_recommendation,
            "confidence_score": result.confidence_score,
            "confirmed_skills": confirmed_skills,
            "knowledge_gaps": knowledge_gaps,
            "soft_skills": SoftSkillsAssessment(
                clarity=result.clarity_score,
                honesty=result.honesty_score,
                engagement=result.engagement_score,
                notes=result.soft_skills_notes
            ),
            "roadmap": result.roadmap,
            "resources": result.resources
        }
        
        # Форматируем отчёт
        report_string = format_report_string(
            feedback, 
            result.verdict_reasoning,
            critical_issues
        )
        
        internal_debate = f"[Reporter]: {result.internal_thought}"
        
        print(f"[Reporter] Мысль: {result.internal_thought}")
        
        return {
            "final_feedback": feedback,
            "final_report_string": report_string,
            "reporter_thought": result.internal_thought,
            "internal_debate": internal_debate,
            "should_end": True
        }
        
    except Exception as e:
        print(f"[Reporter] Ошибка: {e}")
        error_thought = f"Ошибка: {str(e)}"
        return {
            "final_feedback": None,
            "final_report_string": f"Ошибка генерации отчета: {str(e)}",
            "reporter_thought": error_thought,
            "internal_debate": f"[Reporter]: {error_thought}",
            "should_end": True
        }


def format_full_dialogue(turns_log: List[Dict]) -> str:
    """Форматирует ПОЛНУЮ историю диалога с вопросами и ответами."""
    if not turns_log:
        return "Диалог не состоялся"
    
    lines = []
    for turn in turns_log:
        turn_id = turn.get("turn_id", "?")
        question = turn.get("agent_visible_message", "")
        answer = turn.get("user_message", "")
        
        lines.append(f"--- ХОД {turn_id} ---")
        lines.append(f"ВОПРОС ИНТЕРВЬЮЕРА: {question}")
        lines.append(f"ОТВЕТ КАНДИДАТА: {answer}")
        lines.append("")
    
    return "\n".join(lines)


def format_agents_analysis(turns_log: List[Dict]) -> str:
    """Форматирует анализ агентов по каждому ходу."""
    if not turns_log:
        return "Анализ не проводился"
    
    lines = []
    for turn in turns_log:
        turn_id = turn.get("turn_id", "?")
        thoughts = turn.get("internal_thoughts", "")
        
        if thoughts:
            lines.append(f"--- ХОД {turn_id}: Анализ агентов ---")
            # Парсим thoughts по агентам
            for line in thoughts.split("\n"):
                if line.strip():
                    lines.append(f"  {line.strip()}")
            lines.append("")
    
    return "\n".join(lines)


def count_unanswered_questions(turns_log: List[Dict]) -> int:
    """Подсчитывает количество вопросов, на которые кандидат не ответил."""
    unanswered = 0
    
    for turn in turns_log:
        thoughts = turn.get("internal_thoughts", "").lower()
        
        # Ищем индикаторы того, что кандидат не ответил на вопрос
        not_answered_indicators = [
            "не ответил на вопрос",
            "не отвечает на вопрос",
            "ушёл от темы",
            "уходит от темы",
            "off-topic",
            "не по теме",
            "не привёл пример",
            "отсутствует пример",
            "не показал",
            "отвлечённый пример"
        ]
        
        for indicator in not_answered_indicators:
            if indicator in thoughts:
                unanswered += 1
                break
    
    return unanswered


def collect_critical_issues(state: InterviewState) -> str:
    """Собирает критические проблемы из всей истории."""
    issues = []
    
    # Галлюцинации
    hallucination_count = state["behavioral_context"].get("hallucination_count", 0)
    if hallucination_count > 0:
        issues.append(f"Галлюцинации: {hallucination_count} случаев")
    
    # Противоречия
    contradiction_count = state["behavioral_context"].get("contradiction_count", 0)
    if contradiction_count > 0:
        issues.append(f"Противоречия: {contradiction_count} случаев")
    
    # Проблемы от Skeptic
    skeptic_issues = state.get("_skeptic_issues")
    if skeptic_issues:
        if isinstance(skeptic_issues, list):
            issues.append(f"Технические проблемы: {'; '.join(skeptic_issues)}")
        else:
            issues.append(f"Технические проблемы: {skeptic_issues}")
    
    # Анализируем turns_log для сбора проблем
    unanswered = count_unanswered_questions(state.get("turns_log", []))
    if unanswered > 0:
        issues.append(f"Вопросов без ответа: {unanswered} из {len(state.get('turns_log', []))}")
    
    return "\n".join(issues) if issues else "Критических проблем не выявлено."


def format_topics_summary(plan: List[Dict]) -> str:
    if not plan:
        return "Темы не были затронуты"
    
    lines = []
    for topic in plan:
        score = topic.get("score")
        score_str = f"{score}/10" if score is not None else "не оценено"
        weak = topic.get("weak_answers", 0)
        weak_str = f" (слабых ответов: {weak})" if weak > 0 else ""
        lines.append(f"- {topic['topic']}: {score_str}{weak_str}")
    
    return "\n".join(lines)


def format_report_string(
    feedback: dict, 
    verdict_reasoning: str,
    critical_issues: str
) -> str:
    """Форматирует отчёт как строку (чистый формат без эмодзи)"""
    
    lines = [
        "Финальный отчет по интервью",
        "",
        "Вердикт:",
        f"  Оценка уровня: {feedback['assessed_grade']}",
        f"  Рекомендация: {feedback['hiring_recommendation']}",
        f"  Уверенность: {feedback['confidence_score']}%",
        f"  Обоснование: {verdict_reasoning}",
    ]
    
    # Критические проблемы
    if critical_issues and critical_issues != "Критических проблем не выявлено.":
        lines.extend(["", "Критические проблемы:"])
        for issue in critical_issues.split("\n"):
            if issue.strip():
                lines.append(f"  - {issue}")
    
    # Hard Skills
    lines.extend(["", "Hard Skills:"])
    
    if feedback["confirmed_skills"]:
        lines.append("  Подтверждённые навыки:")
        for skill in feedback["confirmed_skills"]:
            lines.append(f"    - {skill['topic']}: {skill['score']}/10")
    
    if feedback["knowledge_gaps"]:
        lines.append("  Пробелы в знаниях:")
        for gap in feedback["knowledge_gaps"]:
            lines.append(f"    - {gap['topic']}: {gap['score']}/10")
    
    if not feedback["confirmed_skills"] and not feedback["knowledge_gaps"]:
        lines.append("  (недостаточно данных для оценки)")
    
    # Soft Skills
    lines.extend([
        "",
        "Soft Skills:",
        f"  Ясность изложения: {feedback['soft_skills']['clarity']}/10",
        f"  Честность: {feedback['soft_skills']['honesty']}/10",
        f"  Вовлечённость: {feedback['soft_skills']['engagement']}/10",
    ])
    
    if feedback["soft_skills"].get("notes"):
        lines.append(f"  Заметки: {feedback['soft_skills']['notes']}")
    
    # Roadmap
    if feedback.get("roadmap"):
        lines.extend(["", "Рекомендации по развитию:"])
        for i, topic in enumerate(feedback["roadmap"][:5], 1):
            lines.append(f"  {i}. {topic}")
    
    # Resources
    if feedback.get("resources"):
        lines.extend(["", "Полезные ресурсы:"])
        for resource in feedback["resources"][:3]:
            lines.append(f"  - {resource}")
    
    return "\n".join(lines)

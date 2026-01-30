#!/usr/bin/env python3
"""
Multi-Agent Interview Coach - CLI Interface
The Cognitive Council System

ПАЙПЛАЙН:
1. Ввод данных кандидата
2. Кандидат пишет приветствие (ПЕРВЫЙ)
3. Агент отвечает и задает вопрос
4. Цикл продолжается до "стоп"

Запуск:
    python main.py

С параметрами:
    python main.py --name "Алекс" --role "Backend Developer" --grade "Junior" --experience "Django, SQL"

С отладкой:
    python main.py --debug
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Загружаем переменные окружения
from dotenv import load_dotenv
load_dotenv()

# Rich для красивого вывода
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.graph import InterviewCoach
from src.logger import InterviewLogger


def create_console():
    """Создает консоль"""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_styled(console, text: str, style: str = None):
    """Выводит текст с стилем"""
    if console and RICH_AVAILABLE:
        if style:
            console.print(text, style=style)
        else:
            console.print(text)
    else:
        print(text)


def print_panel(console, content: str, title: str = None, border_style: str = "blue"):
    """Выводит панель"""
    if console and RICH_AVAILABLE:
        console.print(Panel(content, title=title, border_style=border_style))
    else:
        print(f"\n{'='*60}")
        if title:
            print(f"  {title}")
            print(f"{'='*60}")
        print(content)
        print(f"{'='*60}\n")


def get_user_input(console, prompt_text: str) -> str:
    """Получает ввод пользователя (однострочный)"""
    if console and RICH_AVAILABLE:
        return Prompt.ask(f"[bold green]{prompt_text}[/bold green]")
    else:
        return input(f"{prompt_text}: ")


def get_multiline_input(console, prompt_text: str) -> str:
    """
    Получает многострочный ввод пользователя.
    Пустая строка (двойной Enter) завершает ввод.
    """
    if console and RICH_AVAILABLE:
        console.print(f"[bold green]{prompt_text}[/bold green] [dim](пустая строка для отправки)[/dim]")
    else:
        print(f"{prompt_text} (пустая строка для отправки):")
    
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                # Пустая строка - завершаем ввод
                if lines:
                    break
                # Если ещё ничего не введено, продолжаем ждать
                continue
            lines.append(line)
        except EOFError:
            break
    
    return "\n".join(lines)


def print_header(console):
    """Выводит заголовок"""
    header = """
╔══════════════════════════════════════════════════════════════════╗
║            🎯 MULTI-AGENT INTERVIEW COACH 🎯                     ║
║                 The Cognitive Council System                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Агенты:                                                          ║
║  • Router   - Классификация интента пользователя                 ║
║  • Skeptic  - Технический анализ (Hard Skills) + Фактчекинг      ║
║  • Empath   - Поведенческий анализ (Soft Skills)                 ║
║  • Planner  - Стратегическое планирование и агрегация            ║
║  • Voice    - Ведение диалога с кандидатом                       ║
║  • Reporter - Генерация финального отчета                        ║
╚══════════════════════════════════════════════════════════════════╝
    """
    if console and RICH_AVAILABLE:
        console.print(header, style="bold cyan")
    else:
        print(header)


def collect_candidate_info(console, args) -> dict:
    """Собирает информацию о кандидате"""
    print_styled(console, "\n📝 Введите данные кандидата:\n", "bold yellow")
    
    name = args.name if args.name else get_user_input(console, "Имя кандидата")
    role = args.role if args.role else get_user_input(console, "Позиция (например: Backend Developer)")
    grade = args.grade if args.grade else get_user_input(console, "Грейд (Junior/Middle/Senior)")
    experience = args.experience if args.experience else get_user_input(console, "Опыт (кратко)")
    
    return {
        "name": name,
        "role": role,
        "grade": grade,
        "experience": experience
    }


def save_interview_log(coach: InterviewCoach, logger: InterviewLogger, scenario_id: int = None) -> str:
    """Сохраняет лог интервью в JSON"""
    session_data = coach.export_session()
    
    if logger.session_file:
        logger.current_session.update(session_data)
        logger._save()
        return str(logger.session_file)
    
    # Fallback
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    filename = logs_dir / f"interview_log_{scenario_id}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    return str(filename)


def run_interview(args):
    """Основной цикл интервью"""
    console = create_console()
    
    # Проверяем API ключ
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        print_styled(console, "\n❌ Ошибка: OPENAI_API_KEY не установлен или некорректен!", "bold red")
        print_styled(console, "Создайте файл .env с вашим API ключом.", "yellow")
        return
    
    print_header(console)
    
    # Собираем данные кандидата
    candidate = collect_candidate_info(console, args)
    
    print_panel(
        console,
        f"Кандидат: {candidate['name']}\n"
        f"Позиция: {candidate['role']}\n"
        f"Грейд: {candidate['grade']}\n"
        f"Опыт: {candidate['experience']}",
        title="📋 Данные кандидата"
    )
    
    # Создаем логгер
    logger = InterviewLogger()
    log_path = logger.start_session("Садреддинов Джавид Ханбаба оглы", args.scenario)
    print_styled(console, f"\n📁 Лог сохраняется в: {log_path}\n", "dim")
    
    # Создаем Interview Coach
    print_styled(console, "\n⏳ Инициализация системы...\n", "yellow")
    
    try:
        coach = InterviewCoach()
        
        # Инициализируем интервью (план создается, но приветствия нет)
        init_message = coach.start_interview(
            name=candidate["name"],
            role=candidate["role"],
            grade=candidate["grade"],
            experience=candidate["experience"]
        )
        
        print_styled(console, f"\n✅ {init_message}\n", "green")
        print_panel(
            console,
            "Система готова к интервью.\n"
            "Кандидат должен представиться первым.\n\n"
            "📝 Многострочный ввод: пишите текст, затем нажмите Enter дважды.\n"
            "🛑 Для завершения введите: 'стоп' или 'достаточно'",
            title="ℹ️ Инструкция",
            border_style="yellow"
        )
        
        # Основной цикл
        while coach.is_interview_active():
            # Получаем многострочный ввод кандидата
            user_input = get_multiline_input(console, "\n👤 Кандидат")
            
            if not user_input.strip():
                print_styled(console, "Пожалуйста, введите сообщение.", "yellow")
                continue
            
            # Обрабатываем сообщение
            print_styled(console, "\n⏳ Агенты анализируют ответ...\n", "dim")
            
            response = coach.process_message(user_input)
            
            # Показываем internal thoughts (в режиме отладки)
            if args.debug:
                state = coach.get_state()
                if state:
                    # Показываем мысли каждого агента
                    agent_thoughts = coach.get_agent_thoughts()
                    thoughts_lines = []
                    
                    if agent_thoughts.get("router"):
                        thoughts_lines.append(f"🔀 [Router]: {agent_thoughts['router']}")
                    if agent_thoughts.get("skeptic"):
                        thoughts_lines.append(f"🔬 [Skeptic]: {agent_thoughts['skeptic']}")
                    if agent_thoughts.get("empath"):
                        thoughts_lines.append(f"💚 [Empath]: {agent_thoughts['empath']}")
                    if agent_thoughts.get("planner"):
                        thoughts_lines.append(f"📋 [Planner]: {agent_thoughts['planner']}")
                    if agent_thoughts.get("voice"):
                        thoughts_lines.append(f"🎤 [Voice]: {agent_thoughts['voice']}")
                    
                    if thoughts_lines:
                        print_panel(
                            console,
                            "\n".join(thoughts_lines),
                            title="🧠 Internal Thoughts (Debug)",
                            border_style="magenta"
                        )
            
            # Логируем ход
            turns = coach.get_turns_log()
            if turns:
                last_turn = turns[-1]
                logger.log_turn(
                    turn_id=last_turn["turn_id"],
                    agent_visible_message=last_turn["agent_visible_message"],
                    user_message=last_turn["user_message"],
                    internal_thoughts=last_turn["internal_thoughts"]
                )
            
            # Показываем ответ интервьюера
            print_panel(console, response, title="🤖 Интервьюер", border_style="blue")
        
        # Интервью завершено
        print_styled(console, "\n✅ Интервью завершено!\n", "bold green")
        
        # Сохраняем финальный фидбэк
        feedback = coach.get_final_feedback()
        if feedback:
            logger.log_final_feedback(feedback)
        
        # Сохраняем лог
        final_log_path = save_interview_log(coach, logger, args.scenario)
        print_styled(console, f"\n📁 Лог сохранен: {final_log_path}\n", "bold green")
        
        # Показываем краткую статистику
        turns_count = len(coach.get_turns_log())
        print_styled(console, f"📊 Всего ходов: {turns_count}", "cyan")
        
    except KeyboardInterrupt:
        print_styled(console, "\n\n⚠️ Интервью прервано пользователем.", "yellow")
        try:
            save_interview_log(coach, logger)
        except:
            pass
    except Exception as e:
        print_styled(console, f"\n❌ Ошибка: {str(e)}", "bold red")
        if args.debug:
            import traceback
            traceback.print_exc()


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Interview Coach - The Cognitive Council System"
    )
    parser.add_argument("--scenario", "-s", type=int, help="ID сценария")
    parser.add_argument("--name", "-n", type=str, help="Имя кандидата")
    parser.add_argument("--role", "-r", type=str, help="Позиция")
    parser.add_argument("--grade", "-g", type=str, choices=["Junior", "Middle", "Senior"], help="Грейд")
    parser.add_argument("--experience", "-e", type=str, help="Опыт")
    parser.add_argument("--debug", "-d", action="store_true", help="Режим отладки (показывать internal thoughts)")
    
    args = parser.parse_args()
    run_interview(args)


if __name__ == "__main__":
    main()

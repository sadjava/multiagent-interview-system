# Multi-Agent Interview System

**Cognitive Council** — мультиагентная система для проведения технических интервью с автоматической оценкой кандидатов.

## Обзор

Система использует 6 специализированных AI-агентов, которые работают совместно для проведения структурированного технического интервью:

| Агент | Роль | Задача |
|-------|------|--------|
| **Router** | Классификатор | Определяет тип сообщения кандидата |
| **Skeptic** | Hard Skills | Оценивает техническую корректность |
| **Empath** | Soft Skills | Оценивает манеру общения |
| **Planner** | Стратег | Управляет ходом интервью |
| **Voice** | Интервьюер | Генерирует вопросы и ответы |
| **Reporter** | Аналитик | Формирует финальный отчёт |

## Архитектура

```
      [ Candidate Info ] 
              |
              v
    +-----------------------+      
    |  Topic Planner Node   | ----+ (Initializes Skill Tree in State)
    +-----------------------+     |
              |                   v
              |          +----------------------------------+
              |          |        SHARED STATE (Memory)     |
              |          |----------------------------------|
              v          | - Skill Tree (Topics & Scores)   |
      ( Start Loop ) <---| - Internal Debate Buffer         |
              |          | - Messages & Turns History       |
              |          +----------------------------------+
              v                   ^              ^
    +-----------------------+     |              |
    |     Input Router      |     |              |
    +----------+------------+     |              |
               |                  |              |
        /------+------\           |              |
       /               \          |              |
 [ Tech Answer ]   [ Q / Offtopic ]              |
     |                 |          |              |
     v                 +----------|--------------+
+-----------+                     |              |
|  SKEPTIC  | (Tech Review)       |              |
+-----------+                     |              |
     |                            |              |
     v                            |              |
+-----------+                     |              |
|  EMPATH   | (Soft Skills)       |              |
+-----------+                     |              |
     |                            |              |
     v                            v              |
+------------------------------------+           |
|         STRATEGIC PLANNER          | ----------+ (Updates State & Logic)
+------------------+-----------------+ 
                   |
         /---------+---------\
    [ Continue ]          [ End / Limit ]
         |                       |
         v                       v
+-------------------+      +--------------------+
|  INTERVIEWER VOICE|      |   FINAL REPORTER   |
+-------------------+      +--------------------+
         |                       |
         v                       v
    ( Next Turn )            [ FINAL JSON ]
```
## Быстрый старт

### 1. Клонирование

```bash
git clone <repository-url>
cd multiagent
```

### 2. Окружение

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3. Конфигурация

```bash
cp env_example.txt .env
```

Отредактируйте `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_MODEL_FAST=gpt-4o-mini
MAX_TURNS=10
```

### 4. Запуск

```bash
# Интерактивный режим
python main.py

# С параметрами
python main.py --name "Иван" --role "Backend Developer" --grade "Middle" --experience "Python, Django, PostgreSQL"

# С отладкой (показывает мысли агентов)
python main.py --debug
```

## Пайплайн интервью

1. **Инициализация** — система генерирует план на основе профиля кандидата
2. **Приветствие** — агент начинает диалог и задаёт первый вопрос
3. **Цикл интервью:**
   - Router классифицирует ответ (answer/question/off_topic/stop)
   - Skeptic оценивает техническую часть
   - Empath оценивает soft skills
   - Planner принимает решение о следующем шаге
   - Voice генерирует ответ/вопрос
4. **Завершение** — Reporter формирует финальный отчёт

## Типы интентов

| Интент | Описание | Действие |
|--------|----------|----------|
| `answer` | Кандидат отвечает на вопрос | Полный анализ (Skeptic + Empath) |
| `question` | Кандидат задаёт встречный вопрос | Быстрый ответ + продолжение |
| `off_topic` | Кандидат ушёл от темы | Вернуть к теме |
| `stop` | Кандидат хочет завершить | Генерация отчёта |

## Структура проекта

```
multiagent/
├── main.py                  # CLI интерфейс
├── requirements.txt         # Зависимости
├── env_example.txt          # Шаблон конфигурации
├── src/
│   ├── state.py             # Типы и Pydantic модели
│   ├── graph.py             # LangGraph пайплайн
│   ├── logger.py            # Логирование сессий
│   ├── agents/
│   │   ├── router.py        # Классификатор интентов
│   │   ├── skeptic.py       # Оценка Hard Skills
│   │   ├── empath.py        # Оценка Soft Skills
│   │   ├── planner.py       # Стратегическое планирование
│   │   ├── voice.py         # Генерация ответов
│   │   └── reporter.py      # Финальный отчёт
│   └── tools/
│       └── semantic_router.py
└── logs/                    # JSON логи интервью
```

## Формат логов

Каждое интервью сохраняется в `logs/interview_{name}_{timestamp}.json`:

```json
{
  "participant_name": "Иван",
  "session_start": "2026-01-30T22:00:00",
  "metadata": {
    "role": "Backend Developer",
    "target_grade": "Middle",
    "experience": "Python, Django, PostgreSQL"
  },
  "turns": [
    {
      "turn_id": 1,
      "agent_visible_message": "Привет, Иван! Расскажите про опыт с Django ORM.",
      "user_message": "Использовал select_related для оптимизации...",
      "internal_thoughts": "[Router]: answer\n[Skeptic]: Хороший ответ, 8/10\n[Empath]: Уверенная речь\n[Planner]: Переходим к следующей теме"
    }
  ],
  "final_feedback": "Финальный отчет по интервью\n\nВердикт:\n  Оценка уровня: Middle\n  Рекомендация: Hire\n  Уверенность: 85%\n..."
}
```

## Агенты

### Skeptic (Hard Skills)

Оценивает техническую корректность ответа:
- **score** (0-10) — общая оценка
- **accuracy** — точный / частично_верный / неверный / галлюцинация
- **depth** — поверхностный / достаточный / глубокий / экспертный

### Empath (Soft Skills)

Оценивает манеру общения (НЕ техническую часть):
- **clarity** (1-10) — ясность изложения
- **honesty** (1-10) — честность, признание незнания
- **engagement** — вовлечённость в диалог
- **stress_level** — уровень стресса

### Planner

Управляет стратегией интервью:
- Переход между темами (после 2 вопросов по теме)
- Выбор протокола (standard/rescue/speedrun/stress_test)
- Директивы для Voice

### Reporter

Генерирует финальный отчёт:
- Оценка уровня (Junior/Middle/Senior)
- Рекомендация (Hire/No Hire/Strong Hire)
- Hard Skills с оценками
- Soft Skills с оценками
- Roadmap для развития

## Конфигурация моделей

Система поддерживает разные модели для разных агентов:

| Агент | Модель | Параметры |
|-------|--------|-----------|
| Router | OPENAI_MODEL_FAST | temperature=0, reasoning_effort=minimal |
| Skeptic | OPENAI_MODEL_FAST | temperature=0.1, reasoning_effort=low |
| Empath | OPENAI_MODEL_FAST | temperature=0.3, reasoning_effort=low |
| Planner | OPENAI_MODEL_FAST | temperature=0.3, reasoning_effort=low |
| Voice | OPENAI_MODEL_FAST | temperature=0.7, reasoning_effort=low |
| Reporter | OPENAI_MODEL | temperature=0.2, reasoning_effort=low |

## Многострочный ввод

В интерактивном режиме поддерживается многострочный ввод:
- Пишите текст
- Нажмите Enter дважды (пустая строка) для отправки

## Зависимости

```
langchain>=0.3.0
langchain-openai>=0.2.0
langgraph>=0.2.0
python-dotenv>=1.0.0
pydantic>=2.0.0
rich>=13.0.0
```

## Лицензия

MIT License

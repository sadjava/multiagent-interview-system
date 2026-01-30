"""
Semantic Router - классификатор интента на основе эмбеддингов.
Использует OpenAI text-embedding-3-small для семантического сравнения.
Быстрее и дешевле, чем вызов LLM для классификации.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI


# ============================================================
# Примеры для каждого интента (расширенные и разнообразные)
# ============================================================

INTENT_EXAMPLES: Dict[str, List[str]] = {
    "answer": [
        # Правильные технические ответы
        "Список в Python - это изменяемая упорядоченная коллекция элементов",
        "Django ORM позволяет работать с базой данных через Python объекты",
        "REST API использует HTTP методы GET, POST, PUT, DELETE",
        "Декоратор - это функция, которая принимает другую функцию и расширяет её поведение",
        "GIL в Python блокирует выполнение нескольких потоков одновременно",
        "Миграции в Django создаются командой makemigrations и применяются через migrate",
        "Класс в Python создается с помощью ключевого слова class",
        "SQL JOIN объединяет строки из двух или более таблиц",
        "Асинхронность в Python реализуется через async/await",
        "Git commit фиксирует изменения в локальном репозитории",
        
        # Ответы на технические вопросы (развёрнутые)
        "QuerySet — это ленивый объект, запрос выполняется только при обращении к данным. filter() для фильтрации, select_related() для JOIN",
        "Управление памятью в Python: reference counting как основной механизм, плюс GC для циклических ссылок",
        "Для масштабирования использую CQRS, Event Sourcing, Saga pattern, message brokers как Kafka или RabbitMQ",
        "asyncio использует event loop с epoll/kqueue. Корутины — это генераторы с протоколом __await__",
        "Singleton можно реализовать через модуль, __new__, метакласс или декоратор. Предпочитаю DI",
        
        # Спорные/неверные технические утверждения (тоже answer!)
        "Я читал на Хабре, что в Python 4.0 будут новые фичи",
        "Честно говоря, я слышал что циклы for уберут из языка",
        "Насколько я знаю, в новой версии Python добавят нейронные связи",
        "Мне кажется, GIL уже удалили в последней версии Python",
        "По-моему, Django уже устарел и все переходят на FastAPI",
        "Я не учу циклы, потому что их скоро заменят на что-то другое",
        "asyncio — это то же самое что многопоточность, просто синтаксис другой",
        "Threading в Python — это просто заглушка для совместимости",
        "В PEP 9999 написано про удаление GIL",
        
        # Краткие ответы (тоже answer)
        "Список — это коллекция. Можно добавлять и удалять",
        "Словарь — ключ-значение. Да.",
        "Функция — это блок кода. def name(): pass",
        "Ну это такие штуки в квадратных скобках",
        "Используется для хранения данных",
        
        # Ответы кандидата на свои же размышления
        "Ладно, про списки... это изменяемые коллекции",
        "Хорошо, отвечу про декораторы. Это функции-обёртки",
    ],
    
    "question": [
        # Встречные вопросы кандидата о работе
        "А какие задачи будут на испытательном сроке?",
        "Какой стек технологий вы используете?",
        "Вы работаете с микросервисами или монолитом?",
        "Какая команда будет у меня на проекте?",
        "Есть ли возможность удаленной работы?",
        "Какие перспективы роста в компании?",
        "Используете ли вы Agile/Scrum методологию?",
        "Какой график работы предполагается?",
        "Есть ли код-ревью в команде?",
        "Какие инструменты используете для CI/CD?",
        "Можно узнать про зарплатную вилку?",
        "Какой у вас процесс онбординга?",
        "Работаете ли вы с облачными сервисами?",
        "Есть ли обучение или менторство для джунов?",
        "Какая версия Python используется в проекте?",
        "Как часто релизите в продакшен?",
        "Сколько человек в команде бэкенда?",
        "Какой размер кодовой базы?",
        "Используете PostgreSQL или MySQL?",
        "Какие задачи считаются приоритетными сейчас?",
        
        # КРИТИЧНО: Просьбы повторить/уточнить вопрос (НЕ stop!)
        "Можете повторить вопрос?",
        "Извините, можете повторить вопрос? Я немного растерялась",
        "Не понял вопрос, можете уточнить?",
        "Простите, не расслышал. Повторите, пожалуйста",
        "Можно ещё раз вопрос?",
        "Секунду, соберусь с мыслями... Про списки, да?",
        "Подождите, дайте подумать... О чём был вопрос?",
        "Извините, можно чуть больше контекста к вопросу?",
        "Хм, не совсем понял. Вы спрашиваете про...?",
        "Можете переформулировать вопрос?",
        
        # Бытовые вопросы (тоже question, не off_topic)
        "А у вас в офисе есть кофемашина?",
        "Есть ли PlayStation в офисе?",
        "Какой у вас офис, open space?",
    ],
    
    "off_topic": [
        # Оффтоп - утверждения, НЕ связанные с интервью
        "Какая сегодня хорошая погода",
        "Вчера смотрел интересный фильм",
        "Люблю играть в футбол по выходным",
        "Мой кот вчера разбил вазу",
        "Хочу в отпуск на море",
        "Недавно был на концерте",
        "Сегодня вкусно пообедал в кафе",
        "Мне нравится путешествовать по Европе",
        "Читаю сейчас интересную книгу",
        "Планирую купить новую машину",
        "Вчера готовил пиццу дома",
        "У меня день рождения на следующей неделе",
        "Занимаюсь йогой каждое утро",
        "Смотрели вчера матч? Отличная игра была!",
        "Погода испортилась, дождь идет",
        "Люблю пить кофе по утрам",
        "На выходных ездил на дачу",
        "Слушаю подкасты в дороге",
        "Недавно переехал в новую квартиру",
        "Хочу завести собаку",
        # Попытки увести разговор
        "Кстати, а вы смотрели вчера футбол? Наши выиграли!",
        "Программирование как спорт — нужна практика",
        "Слышал, в IT-компаниях так принято — PlayStation в офисе",
    ],
    
    "stop": [
        # ЯВНЫЕ команды завершения интервью
        "Стоп",
        "Стоп игра",
        "Хватит, давай фидбэк",
        "Закончим интервью",
        "Достаточно, можно завершить",
        "Стоп интервью",
        "Давай заканчивать",
        "Хочу получить фидбэк",
        "Прекращаем собеседование",
        "Стоп, давай результаты",
        "Завершаем, жду обратную связь",
        "Достаточно вопросов, давай итоги",
        "Останови интервью",
        "Хватит вопросов",
        "Закончим на этом",
        "Давай фидбэк по интервью",
        "Стоп, хочу услышать оценку",
        "Заканчиваем собеседование",
        "Всё, достаточно, давай результат",
        "Окей, заканчиваем",
        "Всё.",
        "Думаю, достаточно. Жду обратную связь",
        # Вежливые завершения
        "Спасибо за интервью! Можем завершить?",
        "Спасибо, было интересно! Заканчиваем интервью",
        "Хватит вопросов, давайте результаты!",
    ],
}


@dataclass
class RouteResult:
    """Результат классификации интента"""
    intent: str
    confidence: float
    internal_thought: str


class SemanticRouter:
    """
    Semantic Router для классификации интентов пользователя.
    Использует OpenAI embeddings для семантического сравнения.
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Инициализация роутера.
        
        Args:
            model: Модель эмбеддингов OpenAI
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Кэш эмбеддингов примеров
        self._example_embeddings: Dict[str, np.ndarray] = {}
        self._is_initialized = False
    
    def initialize(self):
        """
        Инициализирует эмбеддинги примеров.
        Вызывается один раз при первом использовании.
        """
        if self._is_initialized:
            return
        
        print("[SemanticRouter] Инициализация эмбеддингов примеров...")
        
        for intent, examples in INTENT_EXAMPLES.items():
            # Получаем эмбеддинги для всех примеров интента
            embeddings = self._get_embeddings(examples)
            self._example_embeddings[intent] = embeddings
            print(f"[SemanticRouter] {intent}: {len(examples)} примеров загружено")
        
        self._is_initialized = True
        print("[SemanticRouter] Инициализация завершена")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Получает эмбеддинги для списка текстов.
        
        Args:
            texts: Список текстов
            
        Returns:
            Numpy array эмбеддингов (N x dim)
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Получает эмбеддинг для одного текста.
        
        Args:
            text: Текст
            
        Returns:
            Numpy array эмбеддинга (1 x dim)
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        
        return np.array(response.data[0].embedding)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Вычисляет косинусное сходство между вектором и матрицей.
        
        Args:
            a: Вектор запроса (dim,)
            b: Матрица примеров (N x dim)
            
        Returns:
            Массив сходств (N,)
        """
        # Нормализуем векторы
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        # Косинусное сходство
        return np.dot(b_norm, a_norm)
    
    def route(
        self, 
        message: str, 
        turn_id: int = 0,
    ) -> RouteResult:
        """
        Классифицирует интент сообщения.
        
        Args:
            message: Сообщение пользователя
            turn_id: Номер хода (для контекста)
            check_suspicious: Не используется (для совместимости)
            
        Returns:
            RouteResult с интентом и метаданными
        """
        # Инициализируем при первом вызове
        if not self._is_initialized:
            self.initialize()

        # Получаем эмбеддинг сообщения
        message_embedding = self._get_embedding(message)
        
        # Вычисляем сходство с каждым интентом
        intent_scores: Dict[str, Tuple[float, float]] = {}  # intent -> (max_score, avg_top3)
        
        for intent, example_embeddings in self._example_embeddings.items():
            similarities = self._cosine_similarity(message_embedding, example_embeddings)
            
            # Берём максимальное сходство и среднее по топ-3
            max_sim = float(np.max(similarities))
            top3_avg = float(np.mean(np.sort(similarities)[-3:]))
            
            intent_scores[intent] = (max_sim, top3_avg)
        
        # Выбираем интент с наибольшим средним по топ-3
        best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x][1])
        best_score = intent_scores[best_intent][1]
        
        # Формируем internal_thought
        scores_summary = ", ".join([
            f"{intent}: {scores[1]:.3f}" 
            for intent, scores in sorted(intent_scores.items(), key=lambda x: -x[1][1])
        ])
        
        internal_thought = (
            f"Семантический анализ: {best_intent} (уверенность: {best_score:.2f}). "
        )
        
        return RouteResult(
            intent=best_intent,
            confidence=best_score,
            internal_thought=internal_thought
        )

# Глобальный инстанс роутера (lazy initialization)
_router_instance: Optional[SemanticRouter] = None

def get_semantic_router() -> SemanticRouter:
    """
    Возвращает глобальный инстанс SemanticRouter.
    Создаётся при первом вызове.
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = SemanticRouter()
    return _router_instance


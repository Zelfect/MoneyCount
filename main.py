import asyncio
import json
import logging
import math
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from aiohttp import ClientError, ClientSession, ClientTimeout, web


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("finance_bot")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "gemini-3.5-flash-lite").strip()
DB_PATH = os.getenv("DB_PATH", "finance.db")
CURRENCY = "₸"
MAX_REPORT_OPERATIONS = 40
MAX_REPORT_CHARS = 3900
MAX_TRANSACTION_AMOUNT = 1_000_000_000_000_000

dp = Dispatcher()
ai_session: ClientSession | None = None

MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📊 Отчет за сегодня"), KeyboardButton(text="📅 Отчет за месяц")],
        [KeyboardButton(text="🤖 ИИ-анализ трат"), KeyboardButton(text="🗑 Удалить последнюю запись")],
    ],
    resize_keyboard=True,
)

FAST_ENTRY_RE = re.compile(r"^([+-])\s*(\d+(?:[.,]\d{1,2})?)\s+(.+?)\s*$")

QUICK_CATEGORIES = {
    "Еда": ("еда", "продукт", "магазин", "донер", "кофе", "обед", "ужин", "завтрак", "тамақ"),
    "Транспорт": ("такси", "автобус", "метро", "бензин", "транспорт", "жол", "жанармай"),
    "Жилье": ("аренд", "квартир", "коммун", "үй", "пәтер"),
    "Здоровье": ("аптек", "врач", "лекарств", "дәрігер", "дәрі"),
    "Развлечения": ("кино", "игра", "концерт", "развлеч", "ойын"),
    "Зарплата": ("зарплат", "аванс", "жалақы"),
}


class AIServiceError(Exception):
    """Raised when the model response cannot be used."""


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL,
                    date TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_user_date "
                "ON transactions(user_id, date)"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add(
        self, user_id: int, amount: float, category: str, description: str
    ) -> int:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO transactions (user_id, amount, category, description, date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, amount, category, description, now),
            )
            return int(cursor.lastrowid)

    def list_for_period(self, user_id: int, prefix: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT id, amount, category, description, date
                    FROM transactions
                    WHERE user_id = ? AND date LIKE ?
                    ORDER BY date DESC, id DESC
                    """,
                    (user_id, f"{prefix}%"),
                )
            )

    def recent(self, user_id: int, days: int = 90, limit: int = 100) -> list[sqlite3.Row]:
        cutoff = (datetime.now().astimezone() - timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT id, amount, category, description, date
                    FROM transactions
                    WHERE user_id = ? AND date >= ?
                    ORDER BY date DESC, id DESC
                    LIMIT ?
                    """,
                    (user_id, cutoff, limit),
                )
            )

    def get_latest(self, user_id: int) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT id, amount, category, description, date
                FROM transactions WHERE user_id = ?
                ORDER BY date DESC, id DESC LIMIT 1
                """,
                (user_id,),
            ).fetchone()

    def current_balance(self, user_id: int) -> float:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(amount), 0) AS balance "
                "FROM transactions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return float(row["balance"])

    def delete(self, user_id: int, transaction_id: int) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM transactions
                WHERE user_id = ? AND id = ? AND id = (
                    SELECT id FROM transactions WHERE user_id = ?
                    ORDER BY date DESC, id DESC LIMIT 1
                )
                """,
                (user_id, transaction_id, user_id),
            )
            return cursor.rowcount == 1


database: Database | None = None


def format_amount(amount: float) -> str:
    return f"{amount:,.2f}".replace(",", " ").rstrip("0").rstrip(".")


def quick_category(description: str, is_income: bool) -> str:
    if is_income:
        lowered = description.casefold()
        if any(keyword in lowered for keyword in QUICK_CATEGORIES["Зарплата"]):
            return "Зарплата"
        return "Доход"
    lowered = description.casefold()
    for category, keywords in QUICK_CATEGORIES.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "Другое"


def parse_fast_entry(text: str) -> tuple[float, str, str] | None:
    match = FAST_ENTRY_RE.fullmatch(text.strip())
    if not match:
        return None
    sign, raw_amount, description = match.groups()
    amount = float(raw_amount.replace(",", "."))
    description = description.strip()
    if (
        not math.isfinite(amount)
        or amount <= 0
        or amount > MAX_TRANSACTION_AMOUNT
        or not description
    ):
        return None
    signed_amount = amount if sign == "+" else -amount
    return signed_amount, quick_category(description, sign == "+"), description[:200]


def require_database() -> Database:
    if database is None:
        raise RuntimeError("База данных не инициализирована")
    return database


def get_ai_session() -> ClientSession:
    if ai_session is None or ai_session.closed:
        raise AIServiceError(
            "ИИ не настроен. Задайте AI_API_KEY в переменных окружения."
        )
    return ai_session


async def generate_content(prompt: str, *, json_output: bool = False) -> str:
    session = get_ai_session()
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    if json_output:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{AI_MODEL}:generateContent"
        )
        async with session.post(url, json=payload) as response:
            if response.status >= 400:
                details = (await response.text())[:500]
                logger.error("Gemini API returned HTTP %s: %s", response.status, details)
                raise AIServiceError("Gemini API отклонил запрос.")
            data = await response.json()
    except AIServiceError:
        raise
    except (ClientError, asyncio.TimeoutError, json.JSONDecodeError) as error:
        logger.exception("Gemini request failed")
        raise AIServiceError("Не удалось получить ответ от ИИ.") from error

    candidates = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(candidates, list) or not candidates:
        raise AIServiceError("ИИ не вернул ответ.")
    if not isinstance(candidates[0], dict):
        raise AIServiceError("ИИ вернул ответ в неподдерживаемом формате.")
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise AIServiceError("ИИ вернул ответ в неподдерживаемом формате.")
    text = "".join(
        part["text"]
        for part in parts
        if isinstance(part, dict)
        and part.get("thought") is not True
        and isinstance(part.get("text"), str)
    ).strip()
    if not text:
        raise AIServiceError("ИИ вернул пустой ответ.")
    return text


async def classify_message(text: str) -> dict[str, Any]:
    prompt = f"""
Ты помогаешь вести личный финансовый учет. Определи смысл сообщения на русском или казахском.
Верни только JSON со следующими полями:
{{"kind":"transaction|question|other","type":"expense|income|null","amount":number|null,
"category":string|null,"description":string|null}}

Если это запись операции, amount укажи положительным числом без знака, type — expense или income.
Если это вопрос о финансах, kind=question. Если это не операция и не вопрос, kind=other.
Не придумывай сумму. Категорию назови коротко на языке пользователя. Описание сохрани кратко.
Сообщение пользователя: {json.dumps(text, ensure_ascii=False)}
""".strip()
    raw = await generate_content(prompt, json_output=True)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AIServiceError("ИИ вернул некорректный формат данных.") from error
    if not isinstance(result, dict):
        raise AIServiceError("ИИ вернул некорректный формат данных.")

    kind = result.get("kind")
    if not isinstance(kind, str) or kind not in {"transaction", "question", "other"}:
        raise AIServiceError("ИИ не смог определить тип сообщения.")
    if kind == "transaction":
        if isinstance(result.get("amount"), bool):
            raise AIServiceError("ИИ не смог определить сумму операции.")
        try:
            amount = float(result.get("amount"))
        except (TypeError, ValueError) as error:
            raise AIServiceError("ИИ не смог определить сумму операции.") from error
        transaction_type = result.get("type")
        category = result.get("category")
        description = result.get("description")
        if (
            not math.isfinite(amount)
            or amount <= 0
            or amount > MAX_TRANSACTION_AMOUNT
            or not isinstance(transaction_type, str)
            or transaction_type not in {"expense", "income"}
            or not isinstance(category, str)
            or not category.strip()
            or not isinstance(description, str)
            or not description.strip()
        ):
            raise AIServiceError("ИИ вернул неполные данные операции.")
        result["amount"] = amount
        result["category"] = category.strip()[:60]
        result["description"] = description.strip()[:200]
    return result


def make_report(rows: list[sqlite3.Row], title: str, current_balance: float) -> str:
    if not rows:
        return (
            f"{title}\nЗа этот период записей пока нет.\n"
            f"Текущий баланс: {format_amount(current_balance)} {CURRENCY}"
        )

    incomes = sum(float(row["amount"]) for row in rows if row["amount"] > 0)
    expenses = -sum(float(row["amount"]) for row in rows if row["amount"] < 0)
    lines = [
        title,
        f"Доходы: {format_amount(incomes)} {CURRENCY}",
        f"Расходы: {format_amount(expenses)} {CURRENCY}",
        f"Текущий баланс: {format_amount(current_balance)} {CURRENCY}",
        "",
        "Операции:",
    ]
    shown = 0
    for row in rows[:MAX_REPORT_OPERATIONS]:
        date = str(row["date"]).replace("T", " ")[:16]
        description = str(row["description"]).replace("\n", " ")[:100]
        line = (
            f"{date} — {format_amount(float(row['amount']))} {CURRENCY} "
            f"({str(row['category'])[:40]}): {description}"
        )
        if len("\n".join(lines)) + len(line) + 1 > MAX_REPORT_CHARS - 80:
            break
        lines.append(line)
        shown += 1
    if shown < len(rows):
        lines.append(f"Показано {shown} из {len(rows)} операций.")
    return "\n".join(lines)


async def send_report(message: Message, prefix: str, title: str) -> None:
    user_database = require_database()
    rows = user_database.list_for_period(message.from_user.id, prefix)
    balance = user_database.current_balance(message.from_user.id)
    await message.answer(make_report(rows, title, balance))


async def save_transaction(
    message: Message, amount: float, category: str, description: str
) -> None:
    require_database().add(message.from_user.id, amount, category, description)
    action = "Доход" if amount > 0 else "Расход"
    await message.answer(
        f"✅ {action} записан: {format_amount(amount)} {CURRENCY}\n"
        f"Категория: {category}\nОписание: {description}",
        reply_markup=MENU,
    )


async def run_analysis(message: Message, question: str) -> None:
    if ai_session is None:
        await message.answer(
            "ИИ-анализ недоступен: задайте AI_API_KEY в переменных окружения."
        )
        return
    rows = require_database().recent(message.from_user.id)
    if not rows:
        await message.answer(
            "За последние 90 дней операций нет. Сначала добавьте траты или доходы."
        )
        return

    history = [
        {
            "date": row["date"],
            "amount": float(row["amount"]),
            "category": row["category"],
            "description": row["description"],
        }
        for row in rows
    ]
    prompt = f"""
Ты — краткий и доброжелательный финансовый аналитик. Проанализируй только эти операции
одного пользователя за последние 90 дней. Отрицательные суммы — расходы, положительные — доходы.
Отвечай на языке вопроса (русском или казахском), конкретно и без обещаний гарантированной экономии.
Не предлагай инвестиции.
Вопрос пользователя: {json.dumps(question, ensure_ascii=False)}
Операции (JSON): {json.dumps(history, ensure_ascii=False)}
""".strip()
    await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        answer = await generate_content(prompt)
    except AIServiceError:
        logger.exception("Could not generate financial analysis")
        await message.answer("Не удалось выполнить ИИ-анализ. Попробуйте позже.")
        return
    await message.answer(answer[:3500])


@dp.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer(
        "Привет! Я помогу вести личный учет финансов.\n\n"
        "Отправьте операцию обычным текстом: «Потратил 1500 на такси» или "
        "«Пришла зарплата 60000».\n"
        "Быстрый формат без ИИ: -1500 такси или +60000 зарплата.\n"
        "Можно задать вопрос о финансах — я отвечу на основе вашей истории.\n\n"
        "Используйте кнопки меню или команды /today и /month.",
        reply_markup=MENU,
    )


@dp.message(Command("today"))
async def today_handler(message: Message) -> None:
    await send_report(
        message, datetime.now().astimezone().date().isoformat(), "📊 Отчет за сегодня"
    )


@dp.message(Command("month"))
async def month_handler(message: Message) -> None:
    await send_report(
        message, datetime.now().astimezone().strftime("%Y-%m"), "📅 Отчет за месяц"
    )


@dp.message(F.text == "📊 Отчет за сегодня")
async def today_button_handler(message: Message) -> None:
    await today_handler(message)


@dp.message(F.text == "📅 Отчет за месяц")
async def month_button_handler(message: Message) -> None:
    await month_handler(message)


@dp.message(F.text == "🤖 ИИ-анализ трат")
async def analysis_button_handler(message: Message) -> None:
    await run_analysis(message, "Проанализируй мои расходы и дай краткие советы.")


@dp.message(F.text == "🗑 Удалить последнюю запись")
async def delete_button_handler(message: Message) -> None:
    row = require_database().get_latest(message.from_user.id)
    if row is None:
        await message.answer("У вас пока нет записей для удаления.")
        return
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Удалить",
                    callback_data=f"delete:{row['id']}",
                ),
                InlineKeyboardButton(text="Отмена", callback_data="delete:cancel"),
            ]
        ]
    )
    await message.answer(
        f"Удалить последнюю запись: {format_amount(float(row['amount']))} {CURRENCY} "
        f"({row['category']}: {row['description']})?",
        reply_markup=keyboard,
    )


@dp.callback_query(F.data.startswith("delete:"))
async def delete_callback_handler(callback: CallbackQuery) -> None:
    if not isinstance(callback.message, Message):
        await callback.answer("Сообщение для удаления недоступно.", show_alert=True)
        return
    if callback.data == "delete:cancel":
        await callback.message.edit_text("Удаление отменено.")
        await callback.answer()
        return
    try:
        transaction_id = int(callback.data.split(":", maxsplit=1)[1])
    except (ValueError, IndexError):
        await callback.answer("Некорректный запрос.", show_alert=True)
        return
    deleted = require_database().delete(callback.from_user.id, transaction_id)
    if deleted:
        await callback.message.edit_text("✅ Последняя запись удалена.")
    else:
        await callback.message.edit_text(
            "Запись не найдена, уже удалена или больше не является последней."
        )
    await callback.answer()


@dp.message(F.text)
async def text_message_handler(message: Message) -> None:
    text = message.text.strip()
    fast_entry = parse_fast_entry(text)
    if fast_entry is not None:
        await save_transaction(message, *fast_entry)
        return

    if ai_session is None:
        await message.answer(
            "Для обычного текста и ИИ-вопросов задайте AI_API_KEY. "
            "Пока можно записывать операции быстрым форматом, например: -1500 такси."
        )
        return

    await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        result = await classify_message(text[:1500])
    except AIServiceError:
        logger.exception("Could not process user message")
        await message.answer("Не удалось обработать сообщение через ИИ. Попробуйте позже.")
        return

    if result["kind"] == "transaction":
        amount = result["amount"]
        if result["type"] == "expense":
            amount = -amount
        await save_transaction(
            message, amount, result["category"], result["description"]
        )
    elif result["kind"] == "question":
        await run_analysis(message, text[:1500])
    else:
        await message.answer(
            "Не распознал операцию или вопрос. Примеры: «Потратил 1500 на такси» "
            "или «Куда ушло больше всего денег за неделю?»"
        )


async def health_check(_: web.Request) -> web.Response:
    return web.Response(text="OK")


async def main() -> None:
    global ai_session, database
    if not BOT_TOKEN:
        raise RuntimeError("Не задана обязательная переменная окружения BOT_TOKEN")
    database = Database(DB_PATH)
    if AI_API_KEY:
        ai_session = ClientSession(
            headers={"x-goog-api-key": AI_API_KEY},
            timeout=ClientTimeout(total=60),
        )
    else:
        logger.warning("AI_API_KEY is not set; AI features are disabled")

    bot = Bot(token=BOT_TOKEN)
    app = web.Application()
    app.router.add_get("/", health_check)
    app.router.add_get("/health", health_check)
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        port = int(os.getenv("PORT", "8080"))
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        logger.info("Health server listening on port %s", port)
        await dp.start_polling(bot)
    finally:
        try:
            await runner.cleanup()
        finally:
            try:
                await bot.session.close()
            finally:
                if ai_session is not None and not ai_session.closed:
                    await ai_session.close()


if __name__ == "__main__":
    asyncio.run(main())

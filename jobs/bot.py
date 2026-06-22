import asyncio
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# === Подключаем корень проекта ===
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from services.predict import get_prediction

# === Логирование ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("bot")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
if not BOT_TOKEN or BOT_TOKEN.startswith("PUT_"):
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN не задан в .env. "
        "Отзови старый через @BotFather (/revoke) и пропиши новый в .env."
    )

USERS_FILE = ROOT / "users.json"
USERS_LOCK = Lock()

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
scheduler = AsyncIOScheduler()

# ==================== КЛАВИАТУРА ====================
keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="AAPL"), KeyboardButton(text="TSLA"), KeyboardButton(text="MSFT"), KeyboardButton(text="GLD")],
        [KeyboardButton(text="^GSPC"), KeyboardButton(text="^IXIC"), KeyboardButton(text="^DJI"), KeyboardButton(text="^RUT")],
        [KeyboardButton(text="📅 5 дней"), KeyboardButton(text="📅 10 дней"), KeyboardButton(text="📅 20 дней")],
        [KeyboardButton(text="Все прогнозы")],
    ],
    resize_keyboard=True,
    persistent=True,
)

# Список инструментов и поддерживаемые горизонты
ALL_SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]
HORIZON_BUTTONS = {"📅 5 дней": 5, "📅 10 дней": 10, "📅 20 дней": 20}


# ==================== USERS PERSISTENCE ====================
def load_users() -> dict:
    with USERS_LOCK:
        if USERS_FILE.exists():
            try:
                with open(USERS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.error("users.json повреждён: %s", e)
        return {}


def save_users(users_dict: dict) -> None:
    with USERS_LOCK:
        tmp = USERS_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(users_dict, f, ensure_ascii=False, indent=2)
        tmp.replace(USERS_FILE)


users = load_users()


def update_user_info(message: types.Message) -> None:
    user = message.from_user
    chat_id = str(message.chat.id)
    if chat_id not in users:
        users[chat_id] = {
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "language_code": user.language_code,
            "subscribed_at": datetime.now().isoformat(),
            "total_predictions": 0,
        }
    users[chat_id]["last_active"] = datetime.now().isoformat()
    save_users(users)


# ====================== HANDLERS ======================
@dp.message(Command("start"))
async def start_handler(message: types.Message):
    update_user_info(message)
    await message.answer(
        "👋 Добро пожаловать!\n\nНажми кнопку ниже, чтобы получить прогноз:",
        reply_markup=keyboard,
    )


@dp.message(Command("me"))
async def me_handler(message: types.Message):
    update_user_info(message)
    chat_id = str(message.chat.id)
    u = users.get(chat_id, {})
    text = (
        f"📋 <b>Твоя информация</b>\n\n"
        f"🆔 ID: <code>{chat_id}</code>\n"
        f"👤 Имя: {u.get('first_name', '—')}\n"
        f"📛 Username: @{u.get('username', '—')}\n"
        f"🌍 Язык: {u.get('language_code', '—')}\n"
        f"📅 Подписка: {(u.get('subscribed_at') or '—')[:10]}\n"
        f"🕒 Последняя активность: {(u.get('last_active') or '—')[:16]}\n"
        f"📊 Запросов: {u.get('total_predictions', 0)}"
    )
    await message.answer(text, parse_mode="HTML")


def _format_prediction(result: dict, lang: str = "ru") -> str:
    hd = result.get("horizon_days", 5)
    if lang == "ru":
        return (
            f"📈 <b>{result['name_ru']} ({result['symbol']})</b>\n"
            f"📅 {result['asof_date']} • горизонт {hd} дней\n\n"
            f"🔹 Рекомендация: <b>{result['recommendation_ru']}</b>\n"
            f"🔹 Уверенность: {result['confidence_ru']}\n"
            f"🔹 Риск: {result['risk_label_ru']}\n\n"
            f"📊 Вероятность роста ({hd}д): <b>{result['p_up']:.1%}</b>\n"
            f"📊 Волатильность ({hd}д): <b>{result['vol_pred']:.2%}</b>\n\n"
            f"🛡️ {result['risk_summary_ru']}"
        )
    return (
        f"📈 <b>{result.get('name_ru', result['symbol'])} ({result['symbol']})</b>\n"
        f"📅 {result['asof_date']}\n\n"
        f"🔹 Recommendation: <b>{result['recommendation_en']}</b>\n"
        f"🔹 Confidence: {result['confidence_en']}\n"
        f"🔹 Risk: {result['risk_label_en']}\n\n"
        f"📊 P(up): <b>{result['p_up']:.1%}</b>\n"
        f"📊 Volatility: <b>{result['vol_pred']:.2%}</b>\n\n"
        f"🛡️ {result['risk_summary_en']}"
    )


@dp.message(F.text.in_(set(HORIZON_BUTTONS.keys())))
async def set_horizon(message: types.Message):
    """Сохраняет выбранный пользователем горизонт прогноза."""
    update_user_info(message)
    chat_id = str(message.chat.id)
    h = HORIZON_BUTTONS[message.text]
    users[chat_id]["horizon"] = h
    save_users(users)
    await message.answer(f"✅ Горизонт прогноза: <b>{h} дней</b>. Теперь выберите инструмент.", parse_mode="HTML")


@dp.message(F.text.in_(set(ALL_SYMBOLS) | {"Все прогнозы"}))
async def quick_predict(message: types.Message):
    update_user_info(message)
    chat_id = str(message.chat.id)
    users[chat_id]["total_predictions"] = users[chat_id].get("total_predictions", 0) + 1
    save_users(users)

    horizon = int(users[chat_id].get("horizon", 5))

    if message.text == "Все прогнозы":
        symbols = ALL_SYMBOLS
        await message.answer(f"🔄 Считаю прогнозы (горизонт {horizon} дней)…")
    else:
        symbols = [message.text]

    for symbol in symbols:
        try:
            # refresh=False — используем закэшированные модели; обновление цен делает daily_update
            result = get_prediction(symbol, horizon=horizon, refresh=False)
            await message.answer(_format_prediction(result, lang="ru"), parse_mode="HTML")
        except Exception as e:
            log.exception("Ошибка прогноза для %s", symbol)
            await message.answer(f"❌ Ошибка по {symbol}: {e}")


# ====================== ЕЖЕДНЕВНАЯ РАССЫЛКА В 19:00 ======================
async def send_daily_forecast():
    if not users:
        return
    log.info("Запуск ежедневной рассылки на %d пользователей", len(users))
    symbols = ALL_SYMBOLS
    for symbol in symbols:
        try:
            result = get_prediction(symbol, horizon=5, refresh=False)
            text = (
                f"🔔 <b>Ежедневный прогноз • {datetime.now().strftime('%d.%m.%Y')}</b>\n\n"
                + _format_prediction(result, lang="ru")
            )
            for chat_id in list(users.keys()):
                try:
                    await bot.send_message(int(chat_id), text, parse_mode="HTML")
                except Exception as e:
                    log.warning("Не удалось отправить %s в чат %s: %s", symbol, chat_id, e)
        except Exception:
            log.exception("Ошибка при подготовке прогноза для %s", symbol)


# ====================== ЗАПУСК ======================
async def main():
    scheduler.add_job(send_daily_forecast, "cron", hour=19, minute=0)
    scheduler.start()

    log.info("🤖 Бот запущен. Пользователей: %d", len(users))
    log.info("✅ Ежедневная рассылка настроена на 19:00")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import logging

from aiogram import Bot, Dispatcher
from config import BOT_TOKEN, MODEL_PATH
from services.inference import PlanGenerator
from handlers import start, plans, price

async def main():
    logging.basicConfig(level=logging.INFO)
    print("Бот запускается...")

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    # Инициализация модели в памяти прямо в bot (доступно в хендлерах)
    try:
        print("Загружаем модель...")
        bot.generator = PlanGenerator(str(MODEL_PATH))
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        bot.generator = None

    dp.include_router(start.router)
    dp.include_router(plans.router)
    dp.include_router(price.router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())


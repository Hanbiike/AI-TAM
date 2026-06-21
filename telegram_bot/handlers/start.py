from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext

from keyboards.main_menu import get_main_menu_kb

router = Router()

welcome_text = """
🏠 <b>Добро пожаловать в AI-TAM!</b>

Этот бот использует технологии искусственного интеллекта для анализа квартир и недвижимости.

<b>Возможности бота:</b>

🏠 <b>Генерация планировок квартир</b>
• Создание новых вариантов планировки на основе контура изображения
• Разделение комнат с визуальной цветовой легендой

💰 <b>Предсказание стоимости квартиры</b>
• Оценка примерной цены недвижимости на основе характеристик <i>(В разработке)</i>

Выберите нужную функцию ниже 👇
"""

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        welcome_text,
        reply_markup=get_main_menu_kb(),
        parse_mode="HTML"
    )

@router.callback_query(F.data == "back_to_main")
async def back_to_main_handler(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    try:
        await callback.message.edit_text(
            welcome_text,
            reply_markup=get_main_menu_kb(),
            parse_mode="HTML"
        )
    except Exception:
        await callback.message.delete()
        await callback.message.answer(
            welcome_text,
            reply_markup=get_main_menu_kb(),
            parse_mode="HTML"
        )
    await callback.answer()


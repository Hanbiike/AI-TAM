from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

def get_main_menu_kb() -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardBuilder()
    keyboard.row(InlineKeyboardButton(text="🏠 Генерация планировки", callback_data="btn_generate_plan"))
    keyboard.row(InlineKeyboardButton(text="💰 Предсказание цены", callback_data="btn_predict_price"))
    return keyboard.as_markup()


from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

def get_price_mode_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.row(InlineKeyboardButton(text="📝 Заполнить по шаблону", callback_data="price_template"))
    kb.row(InlineKeyboardButton(text="💬 Описать квартиру", callback_data="price_describe"))
    kb.row(InlineKeyboardButton(text="🔙 На главную", callback_data="back_to_main"))
    return kb.as_markup()

def get_cancel_price_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.row(InlineKeyboardButton(text="❌ Отменить / На главную", callback_data="back_to_main"))
    return kb.as_markup()


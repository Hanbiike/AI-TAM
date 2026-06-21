from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from pathlib import Path

def get_examples_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    row1 = []
    for i in range(4):
         row1.append(InlineKeyboardButton(text=f"Пример {i}", callback_data=f"show_example_{i}"))
    kb.row(*row1)

    kb.row(InlineKeyboardButton(text="Показать остальные варианты", callback_data="show_more_examples"))
    kb.row(InlineKeyboardButton(text="Загрузить свою картинку", callback_data="upload_own_image"))
    kb.row(InlineKeyboardButton(text="🔙 На главную", callback_data="back_to_main"))
    return kb.as_markup()

def get_more_examples_kb(test_images_dir: Path) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    buttons = []
    for i in range(4, 31):
        if (test_images_dir / f"{i}.png").exists():
            buttons.append(InlineKeyboardButton(text=f"Пример {i}", callback_data=f"show_example_{i}"))

    for i in range(0, len(buttons), 4):
        kb.row(*buttons[i:i+4])

    kb.row(InlineKeyboardButton(text="Загрузить свою картинку", callback_data="upload_own_image"))
    kb.row(InlineKeyboardButton(text="🔙 Назад к выбору", callback_data="btn_generate_plan"))
    return kb.as_markup()

def get_confirm_example_kb(img_idx: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.row(InlineKeyboardButton(text="✅ Сгенерировать план", callback_data=f"generate_example_{img_idx}"))
    kb.row(InlineKeyboardButton(text="🔙 Назад к выбору", callback_data="btn_generate_plan"))
    return kb.as_markup()

def get_post_generation_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.row(InlineKeyboardButton(text="🔙 Назад к схемам", callback_data="btn_generate_plan"))
    kb.row(InlineKeyboardButton(text="🏠 На главную", callback_data="back_to_main"))
    return kb.as_markup()


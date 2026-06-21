import os
import io
from aiogram import Router, F, Bot
from aiogram.types import CallbackQuery, Message, InputMediaPhoto, FSInputFile, BufferedInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from PIL import Image

from config import TEST_IMAGES_DIR
from keyboards.plans import (
    get_examples_kb, get_more_examples_kb,
    get_confirm_example_kb, get_post_generation_kb
)
from utils.legend import LEGEND_TEXT
from services.inference import PlanGenerator

router = Router()

class PlanStates(StatesGroup):
    waiting_for_image = State()

@router.callback_query(F.data == "btn_generate_plan")
async def btn_generate_plan(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.answer()

    examples = [str(TEST_IMAGES_DIR / f"{i}.png") for i in range(4)]

    media_group = []
    for i, path in enumerate(examples):
        if os.path.exists(path):
            media_group.append(InputMediaPhoto(media=FSInputFile(path), caption=f"Пример {i}" if i == 0 else ""))

    if media_group:
        await callback.message.bot.send_media_group(chat_id=callback.message.chat.id, media=media_group)

    await callback.message.bot.send_message(
        chat_id=callback.message.chat.id,
        text="Выберите один из примеров выше для генерации, покажите другие, или загрузите свою:",
        reply_markup=get_examples_kb()
    )

@router.callback_query(F.data == "show_more_examples")
async def show_more_examples(callback: CallbackQuery):
    await callback.answer()
    await callback.message.edit_text(
        text="Доступные дополнительные примеры:",
        reply_markup=get_more_examples_kb(TEST_IMAGES_DIR)
    )

@router.callback_query(F.data.startswith("show_example_"))
async def show_example(callback: CallbackQuery):
    img_idx = callback.data.split("_")[-1]
    img_path = TEST_IMAGES_DIR / f"{img_idx}.png"

    if not img_path.exists():
        await callback.answer("Файл примера не найден", show_alert=True)
        return

    await callback.answer()
    await callback.message.bot.send_photo(
        chat_id=callback.message.chat.id,
        photo=FSInputFile(path=str(img_path)),
        caption=f"Вы выбрали Пример {img_idx}. Сгенерировать планировку для него?",
        reply_markup=get_confirm_example_kb(img_idx)
    )

@router.callback_query(F.data.startswith("generate_example_"))
async def generate_example(callback: CallbackQuery, bot: Bot):
    generator: PlanGenerator = bot.generator
    img_idx = callback.data.split("_")[-1]
    img_path = TEST_IMAGES_DIR / f"{img_idx}.png"

    if not img_path.exists():
        await callback.answer("Файл примера не найден", show_alert=True)
        return

    await callback.answer("Генерируем план...")
    msg = await callback.message.answer("Процесс генерации запущен...")

    try:
        image = Image.open(str(img_path))
        result_img = generator.predict(image)

        bio = io.BytesIO()
        result_img.save(bio, format='PNG')
        bio.seek(0)

        await bot.delete_message(chat_id=msg.chat.id, message_id=msg.message_id)

        success_text = f"🎉 <b>Ваша планировка квартиры успешно сгенерирована!</b>\n\n{LEGEND_TEXT}"
        await bot.send_photo(
            chat_id=callback.message.chat.id,
            photo=BufferedInputFile(bio.read(), filename="generated.png"),
            caption=success_text,
            parse_mode="HTML",
            reply_markup=get_post_generation_kb()
        )
    except Exception as e:
        await bot.edit_message_text(f"Ошибка при генерации: {e}", chat_id=msg.chat.id, message_id=msg.message_id)

@router.callback_query(F.data == "upload_own_image")
async def upload_own_image(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    await state.set_state(PlanStates.waiting_for_image)

    await callback.message.bot.send_message(
        chat_id=callback.message.chat.id,
        text="⚠️ Внимание: нужна красная дверь и черные стены на белом фоне.\n\nПожалуйста, отправьте картинку как фото."
    )

@router.message(PlanStates.waiting_for_image, F.photo)
async def process_user_photo(message: Message, state: FSMContext, bot: Bot):
    await state.clear()
    generator: PlanGenerator = bot.generator
    msg = await message.answer("Фото получено. Начинаю генерацию...")

    try:
        file_id = message.photo[-1].file_id
        file = await bot.get_file(file_id)

        bio_in = io.BytesIO()
        await bot.download(file, destination=bio_in)
        bio_in.seek(0)

        image = Image.open(bio_in)
        result_img = generator.predict(image)

        bio_out = io.BytesIO()
        result_img.save(bio_out, format='PNG')
        bio_out.seek(0)

        await bot.delete_message(chat_id=msg.chat.id, message_id=msg.message_id)

        success_text = f"🎉 <b>Ваша планировка квартиры успешно сгенерирована!</b>\n\n{LEGEND_TEXT}"
        await message.answer_photo(
            photo=BufferedInputFile(bio_out.read(), filename="generated.png"),
            caption=success_text,
            parse_mode="HTML",
            reply_markup=get_post_generation_kb()
        )
    except Exception as e:
        await bot.edit_message_text(f"Ошибка при обработке фото: {e}", chat_id=msg.chat.id, message_id=msg.message_id)

@router.message(PlanStates.waiting_for_image)
async def process_user_photo_wrong(message: Message):
    await message.answer("Пожалуйста, отправьте именно картинку (как фото).")


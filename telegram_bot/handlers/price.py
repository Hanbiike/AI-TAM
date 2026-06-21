from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

from keyboards.price import get_price_mode_kb, get_cancel_price_kb

router = Router()

class PriceStates(StatesGroup):
    waiting_for_rooms = State()
    waiting_for_area = State()
    waiting_for_floor = State()
    waiting_for_series = State()
    waiting_for_status = State()
    waiting_for_year = State()

@router.callback_query(F.data == "btn_predict_price")
async def start_price_prediction(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        "Выберите способ ввода данных для оценки стоимости квартиры:",
        reply_markup=get_price_mode_kb()
    )

@router.callback_query(F.data == "price_describe")
async def price_describe(callback: CallbackQuery):
    await callback.answer("Эта функция находится в доработке 🛠", show_alert=True)

@router.callback_query(F.data == "price_template")
async def price_template_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(PriceStates.waiting_for_rooms)
    await callback.message.edit_text(
        "Шаг 1/6: Введите количество комнат (например, 2):",
        reply_markup=get_cancel_price_kb()
    )

@router.message(PriceStates.waiting_for_rooms)
async def process_rooms(message: Message, state: FSMContext):
    await state.update_data(rooms=message.text)
    await state.set_state(PriceStates.waiting_for_area)
    await message.answer("Шаг 2/6: Введите площадь в квадратных метрах (например, 55.5):\n<i>Помните: в любой момент можно отменить процесс кнопкой ниже.</i>", reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_area)
async def process_area(message: Message, state: FSMContext):
    await state.update_data(area=message.text)
    await state.set_state(PriceStates.waiting_for_floor)
    await message.answer("Шаг 3/6: Введите этаж (например, 3):", reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_floor)
async def process_floor(message: Message, state: FSMContext):
    await state.update_data(floor=message.text)
    await state.set_state(PriceStates.waiting_for_series)
    await message.answer("Шаг 4/6: Укажите серию дома (например, 104-серия, 105-серия, элитка):", reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_series)
async def process_series(message: Message, state: FSMContext):
    await state.update_data(series=message.text)
    await state.set_state(PriceStates.waiting_for_status)
    await message.answer("Шаг 5/6: Укажите состояние квартиры (например, евроремонт, ПСО, без ремонта):", reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_status)
async def process_status(message: Message, state: FSMContext):
    await state.update_data(status=message.text)
    await state.set_state(PriceStates.waiting_for_year)
    await message.answer("Шаг 6/6: Укажите год постройки (например, 2018):", reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_year)
async def process_year(message: Message, state: FSMContext):
    await state.update_data(year=message.text)
    data = await state.get_data()
    await state.clear()

    text = (
        f"✅ <b>Данные собраны!</b>\n\n"
        f"Комнат: {data.get('rooms')}\n"
        f"Площадь: {data.get('area')} м²\n"
        f"Этаж: {data.get('floor')}\n"
        f"Серия: {data.get('series')}\n"
        f"Состояние: {data.get('status')}\n"
        f"Год постройки: {data.get('year')}\n\n"
        f"⏳ <i>Интеграция скоро будет завершена... (Файлы модели price_model.pth и .pkl пока не загружены).</i>"
    )

    await message.answer(text, parse_mode="HTML", reply_markup=get_cancel_price_kb())

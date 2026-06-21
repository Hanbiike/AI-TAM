import asyncio
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

from keyboards.price import get_price_mode_kb, get_cancel_price_kb, get_final_price_kb
from services.p_inference import PricePredictor

router = Router()

# Глобальная инициализация модели
predictor = PricePredictor(
    model_path="price_prediction/price_model.pth",
    context_path="price_prediction/price_context.pkl"
)


class PriceStates(StatesGroup):
    waiting_for_rooms = State()
    waiting_for_area = State()
    waiting_for_floor = State()
    waiting_for_series = State()
    waiting_for_status = State()
    waiting_for_year = State()


def map_user_data_to_features(data: dict) -> dict:
    features = {}
    try:
        features['rooms'] = float(data.get('rooms', 1))
    except (ValueError, TypeError):
        features['rooms'] = 1.0
    try:
        features['area'] = float(str(data.get('area', 50)).replace(',', '.'))
    except (ValueError, TypeError):
        features['area'] = 50.0
    try:
        features['floor'] = float(data.get('floor', 1))
    except (ValueError, TypeError):
        features['floor'] = 1.0
    try:
        year_val = float(data.get('year', 2020))
        features['year'] = year_val
        features['built_year'] = year_val
        features['year_built'] = year_val
    except (ValueError, TypeError):
        pass

    series_text = str(data.get('series', '')).lower()
    if '104' in series_text:
        features['series_104 серия'] = 1.0
        if 'улучш' in series_text: features['series_104 серия улучшенная'] = 1.0
    elif '105' in series_text:
        features['series_105 серия'] = 1.0
        if 'улучш' in series_text: features['series_105 серия улучшенная'] = 1.0
    elif '106' in series_text:
        features['series_106 серия'] = 1.0
        if 'улучш' in series_text: features['series_106 серия улучшенная'] = 1.0
    elif '107' in series_text:
        features['series_107 серия'] = 1.0
    elif '108' in series_text:
        features['series_108 серия'] = 1.0
    elif 'хрущ' in series_text:
        features['series_хрущевка'] = 1.0
    elif 'стал' in series_text:
        features['series_сталинка'] = 1.0
    elif 'элит' in series_text or 'индив' in series_text:
        features['series_индивид. планировка'] = 1.0
    elif 'малосем' in series_text:
        features['series_малосемейка'] = 1.0
    elif 'пентх' in series_text:
        features['series_пентхаус'] = 1.0

    status_text = str(data.get('status', '')).lower()
    if 'евро' in status_text:
        features['status_евроремонт'] = 1.0
    elif 'псо' in status_text or 'самоотд' in status_text:
        features['status_под самоотделку (ПСО)'] = 1.0
        features['status_ПСО'] = 1.0
    elif 'хорош' in status_text or 'космет' in status_text:
        features['status_хорошее'] = 1.0
    elif 'средн' in status_text:
        features['status_среднее'] = 1.0
    elif 'без' in status_text or 'ремонт' in status_text:
        features['status_без ремонта'] = 1.0

    return features


@router.callback_query(F.data == "btn_predict_price")
async def start_price_prediction(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text("Выберите способ ввода данных:", reply_markup=get_price_mode_kb())


@router.callback_query(F.data == "price_template")
async def price_template_start(callback: CallbackQuery, state: FSMContext):
    await state.set_state(PriceStates.waiting_for_rooms)
    await callback.message.edit_text("Шаг 1/6: Введите количество комнат:(например. 3)", reply_markup=get_cancel_price_kb())


@router.message(PriceStates.waiting_for_rooms)
async def process_rooms(message: Message, state: FSMContext):
    if not message.text.isdigit():
        return await message.answer("Введите число цифрами:", reply_markup=get_cancel_price_kb())
    await state.update_data(rooms=message.text)
    await state.set_state(PriceStates.waiting_for_area)
    await message.answer("Шаг 2/6: Введите площадь в м² (например, 55.5):", reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_area)
async def process_area(message: Message, state: FSMContext):
    await state.update_data(area=message.text.replace(',', '.'))
    await state.set_state(PriceStates.waiting_for_floor)
    await message.answer("Шаг 3/6: Введите этаж (например, 5):", reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_floor)
async def process_floor(message: Message, state: FSMContext):
    await state.update_data(floor=message.text)
    await state.set_state(PriceStates.waiting_for_series)
    await message.answer("Шаг 4/6: Укажите серию дома:\n(Примеры: 104 серия, 105 серия, элитка, хрущевка, сталинка)",
                         reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_series)
async def process_series(message: Message, state: FSMContext):
    await state.update_data(series=message.text)
    await state.set_state(PriceStates.waiting_for_status)
    await message.answer("Шаг 5/6: Состояние квартиры:\n(Примеры: евроремонт, ПСО, хорошее, среднее, без ремонта)",
                         reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_status)
async def process_status(message: Message, state: FSMContext):
    await state.update_data(status=message.text)
    await state.set_state(PriceStates.waiting_for_year)
    await message.answer("Шаг 6/6: Введите год постройки (например, 2015):", reply_markup=get_cancel_price_kb())

@router.message(PriceStates.waiting_for_year)
async def process_year(message: Message, state: FSMContext):
    if not message.text.isdigit():
        return await message.answer("Ошибка: введите год цифрами (например, 2021):", reply_markup=get_cancel_price_kb())

    await state.update_data(year=message.text)
    data = await state.get_data()
    await state.clear()

    status_msg = await message.answer("🔄 <i>Провожу оценку...</i>", parse_mode="HTML")
    model_ready_data = map_user_data_to_features(data)

    try:
        predicted_price = await asyncio.to_thread(predictor.predict, model_ready_data)
        text = (
            f"✅ <b>Расчет завершен!</b>\n\n"
            f"<b>Введенные параметры:</b>\n"
            f"• Комнат: {data.get('rooms')}\n"
            f"• Площадь: {data.get('area')} м²\n"
            f"• Этаж: {data.get('floor')}\n"
            f"• Серия: {data.get('series')}\n"
            f"• Состояние: {data.get('status')}\n"
            f"• Год постройки: {data.get('year')}\n\n"
            f"📊 <b>Оценочная стоимость: ${predicted_price:,.2f}</b>"
        )
        try:
            await status_msg.delete()
        except:
            pass

        await message.answer(text, parse_mode="HTML", reply_markup=get_final_price_kb())

    except Exception as e:
        error_text = f"❌ Ошибка: <code>{str(e)}</code>"
        try:
            await status_msg.edit_text(error_text, parse_mode="HTML", reply_markup=get_cancel_price_kb())
        except:
            await message.answer(error_text, parse_mode="HTML", reply_markup=get_cancel_price_kb())
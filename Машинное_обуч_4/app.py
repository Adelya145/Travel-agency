import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any
import pickle
from pydantic import BaseModel
import numpy as np
import warnings
import sklearn

API_BASE_URL = "http://localhost:8000"

currency_dict = {
    1: "TRY",
    2: "GBP",
    3: "EUR",
    4: "USD"
}

currency_symbols = {
    'TRY': '₺',
    'USD': '$',
    'EUR': '€',
    'GBP': '£'
}

# Обновленный словарь типов недвижимости
subtype_mapping = {
    1: "Квартира",
    2: "Частные дома",
    3: "Полное здание"
}

# Словарь типов объявлений
listing_type_mapping = {
    1: "Продажа",
    2: "Аренда"
}


def call_api(endpoint: str, data: Dict = None, method: str = "GET"):
    """Универсальная функция для вызова API"""
    try:
        url = f"{API_BASE_URL}{endpoint}"

        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            return None

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при обращении к API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Неожиданная ошибка: {str(e)}")
        return None


class RealEstateClient:
    def __init__(self):
        self.cities = []
        self.districts = []
        self.counties = []
        self.load_data()

    def load_data(self):
        """Загрузка данных из API"""
        self.cities = call_api("/cities") or []
        self.districts = call_api("/districts") or []
        self.counties = call_api("/counties") or []

    def predict_price(self, features: Dict[str, Any]):
        """Предсказание цены через API"""
        return call_api("/predict/price", features, "POST")

    def predict_subtype(self, features: Dict[str, Any]):
        """Предсказание типа через API"""
        return call_api("/predict/subtype", features, "POST")

    def convert_currency(self, amount: float, from_currency: str, to_currency: str):
        """Конвертация валюты через API"""
        data = {
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency
        }
        return call_api("/convert-currency", data, "POST")


def get_common_features(client, for_price=True):
    """Общая форма для ввода характеристик"""
    col1, col2 = st.columns(2)

    with col1:
        # Тип объявления (Продажа/Аренда)
        listing_type = st.selectbox("Тип объявления",
                                    options=[1, 2],
                                    format_func=lambda x: listing_type_mapping[x])

        size = st.slider("Площадь (m²)", 28.0, 2500.0, 50.0)
        bedroom_count = st.slider("Спальни", 1, 10, 2)
        living_room_count = st.slider("Гостиные", 0, 5, 1)
        building_age_id = st.selectbox("Возраст здания",
                                       options=list(range(1, 15)),
                                       format_func=lambda x: [
                                           "0 лет", "1 год", "2 года", "3 года", "4 года", "5 лет",
                                           "6-10 лет", "11-15 лет", "16-20 лет", "21-25 лет",
                                           "26-30 лет", "31-35 лет", "36-40 лет", "40+ лет"
                                       ][x - 1])

        if not for_price:
            price = st.number_input("Цена (₺)", min_value=0.0, max_value=10000000.0, value=500000.0, step=1000.0)
        else:
            price = 0

    with col2:
        # Города
        city_options = {city['city_id']: city['city_name'] for city in client.cities} if client.cities else {
            1: "Стамбул", 2: "Анкара", 3: "Измир", 4: "Анталья"}
        city_id = st.selectbox("Город", options=list(city_options.keys()),
                               format_func=lambda x: city_options.get(x, f"Город {x}"))

        # Районы
        district_options = {district['district_id']: district['district_name'] for district in
                            client.districts} if client.districts else {1: "Центральный"}
        district_id = st.selectbox("Район", options=list(district_options.keys()),
                                   format_func=lambda x: district_options.get(x, f"Район {x}"))

        # Округи
        county_options = {county['county_id']: county['county_name'] for county in
                          client.counties} if client.counties else {1: "Центральный"}
        county_id = st.selectbox("Округ", options=list(county_options.keys()),
                                 format_func=lambda x: county_options.get(x, f"Округ {x}"))

        heating_type_id = st.selectbox("Отопление",
                                       options=list(range(1, 17)),
                                       format_func=lambda x: [
                                           "Газовый котел", "Кондиционер", "Центральное с счетчиком",
                                           "Центральное", "Газовое отопление", "Угольная печь",
                                           "Теплый пол", "Нет", "Калорифер", "Электрический котел",
                                           "Газовая печь", "Солнечная", "Угольное отопление",
                                           "Геотермальное", "Фанкойл", "Мазутное отопление"
                                       ][x - 1])

    col3, col4 = st.columns(2)
    with col3:
        start_season = st.selectbox("Сезон начала",
                                    options=[1, 2, 3, 4],
                                    format_func=lambda x: ["Зима", "Весна", "Лето", "Осень"][x - 1])

    with col4:
        # Сезон окончания
        end_season = st.selectbox("Сезон окончания",
                                  options=[1, 2, 3, 4],
                                  format_func=lambda x: ["Зима", "Весна", "Лето", "Осень"][x - 1])

    col5, col6, col7 = st.columns(3)
    with col5:
        price_currency_id = st.selectbox("Валюта",
                                         options=[1, 2, 3, 4],
                                         format_func=lambda x: currency_dict[x])
    with col6:
        floor_no_id = st.selectbox("Этаж",
                                   options=list(range(1, 37)),
                                   format_func=lambda x: f"Этаж {x}" if x <= 20 else [
                                       "Высокий вход", "Частный", "Садовый", "Цокольный", "Подвал 1",
                                       "Подвал 2", "Подвал 3", "Подвал 4", "Пентхаус", "Цоколь",
                                       "Комплекс", "20+", "Верхний", "Подвальный", "Терраса", "Антресоль"
                                   ][x - 21])

    with col7:
        # Количество дней на рынке
        tom = st.number_input("Дней на рынке", min_value=0, value=30,
                              help="Количество дней, которые объект находится на рынке")

    features = {
        "size": size,
        "start_season": start_season,
        "end_season": end_season,
        "price_currency_id": price_currency_id,
        "heating_type_id": heating_type_id,
        "building_age_id": building_age_id,
        "city_id": city_id,
        "county_id": county_id,
        "district_id": district_id,
        "bedroom_count": bedroom_count,
        "living_room_count": living_room_count,
        "floor_no_id": floor_no_id,
        "tom": tom,
        "price": price if not for_price else 0,
        "listing_type": listing_type
    }

    if for_price:
        sub_type_id = st.selectbox("Тип недвижимости",
                                   options=[1, 2, 3],
                                   format_func=lambda x: subtype_mapping[x])
        features["sub_type_id"] = sub_type_id
    else:
        features["sub_type_id"] = 1

    return features


def main():
    st.set_page_config(
        page_title="Турецкая недвижимость - предсказание",
        layout="wide"
    )

    client = RealEstateClient()

    # Проверка соединения с API
    health = call_api("/health")
    if health and not all(health.values()):
        st.warning("Некоторые компоненты API не загружены. Возможны ограничения функциональности.")

    st.title("🏠 Предсказатель турецкой недвижимости")

    page = st.sidebar.radio("Выберите вкладку:",
                            ["📈 Предсказание цены", "🏠 Предсказание типа", "ℹ️ О проекте"])

    if page == "📈 Предсказание цены":
        st.header("📈 Предсказание цены недвижимости")

        features = get_common_features(client, for_price=True)
        result_currency = st.selectbox(
            "Валюта для отображения результата",
            options=list(currency_symbols.keys()),
            index=0
        )

        if st.button("🎯 Предсказать цену", type="primary"):
            with st.spinner("Выполняется предсказание..."):
                result = client.predict_price(features)

            if result and "predicted_price" in result:
                selected_currency_id = features["price_currency_id"]
                prediction_currency = currency_dict[selected_currency_id]
                predicted_price = result["predicted_price"]
                listing_type = listing_type_mapping[features["listing_type"]]

                # Конвертация валюты
                conversion = client.convert_currency(predicted_price, prediction_currency, result_currency)
                if conversion and "converted_amount" in conversion:
                    converted_price = conversion["converted_amount"]
                else:
                    converted_price = predicted_price

                symbol = currency_symbols.get(result_currency, '')
                st.success(
                    f"Предсказанная цена ({listing_type.lower()}): **{symbol}{converted_price:,.0f} {result_currency}**")

                if result_currency != prediction_currency:
                    original_symbol = currency_symbols.get(prediction_currency, '')
                    st.info(f"В оригинальной валюте: {original_symbol}{predicted_price:,.0f} {prediction_currency}")

            elif result and "error" in result:
                st.error(f"Ошибка: {result['error']}")

    elif page == "🏠 Предсказание типа":
        st.header("🏠 Предсказание типа недвижимости")

        features = get_common_features(client, for_price=False)

        if st.button("🎯 Предсказать тип", type="primary"):
            with st.spinner("Выполняется предсказание..."):
                result = client.predict_subtype(features)

            if result and "predicted_subtype" in result:
                listing_type = listing_type_mapping[features["listing_type"]]
                st.success(f"Предсказанный тип ({listing_type.lower()}): **{result['predicted_subtype']}**")
                st.info(f"Уверенность предсказания: **{result['confidence']:.1%}**")

            elif result and "error" in result:
                st.error(f"Ошибка: {result['error']}")

    else:
        st.header("ℹ️ О проекте")
        st.markdown("""
        ### Система предсказания недвижимости

        **Архитектура:**
        - 🚀 Серверная часть: FastAPI
        - 🎨 Клиентская часть: Streamlit
        - 📊 Модели машинного обучения

        **Функциональность:**
        - 📈 Предсказание цены недвижимости
        - 🏠 Предсказание типа недвижимости

        **Типы объявлений:**
        - 💰 Продажа
        - 🏠 Аренда

        **Поддерживаемые типы недвижимости:**
        - 🏢 Квартира
        - 🏠 Частные дома  
        - 🏢 Полное здание

        **Поддерживаемые валюты:**
        - TRY (Турецкая лира)
        - USD (Доллар США)
        - EUR (Евро)
        - GBP (Фунт стерлингов)
        
        **Инструкция по применению:**
        - Запустите сервер командой uvicorn main:app --reload
        - Затем запустите клиентскую часть streamlit run app.py
        - Заполните все данные
        - Нажмите кнопку предсказать
        """)


if __name__ == "__main__":
    main()
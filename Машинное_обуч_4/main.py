from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
import os
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings('ignore')

# Отключаем параллельное выполнение для избежания ошибок
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

app = FastAPI(title="Real Estate Prediction API")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PricePredictionRequest(BaseModel):
    size: float
    sub_type_id: int
    start_season: int
    end_season: int
    price_currency_id: int
    heating_type_id: int
    building_age_id: int
    city_id: int
    county_id: int
    district_id: int
    bedroom_count: int
    living_room_count: int
    floor_no_id: int
    tom: int
    listing_type: int


class SubtypePredictionRequest(BaseModel):
    size: float
    start_season: int
    end_season: int
    price_currency_id: int
    heating_type_id: int
    building_age_id: int
    city_id: int
    county_id: int
    district_id: int
    bedroom_count: int
    living_room_count: int
    floor_no_id: int
    tom: int
    price: float
    listing_type: int


class CurrencyConversionRequest(BaseModel):
    amount: float
    from_currency: str
    to_currency: str


# Загрузка моделей
price_model = None
subtype_model = None
price_feature_columns = None
subtype_feature_columns = None


def load_models():
    global price_model, subtype_model, price_feature_columns, subtype_feature_columns

    try:
        # Загрузка модели цены
        try:
            with open('models/best_model_regressor.pkl', 'rb') as f:
                price_model = pickle.load(f)
            if hasattr(price_model, 'feature_names_in_'):
                price_feature_columns = list(price_model.feature_names_in_)
        except:
            price_feature_columns = [
                'listing_type', 'tom', 'size', 'sub_type_id', 'start_season', 'end_season',
                'price_currency_id', 'heating_type_id', 'building_age_id',
                'city_id', 'county_id', 'district_id', 'bedroom_count',
                'living_room_count', 'floor_no_id', 'price'
            ]

        # Загрузка модели типа
        try:
            with open('models/best_model_classification.pkl', 'rb') as f:
                subtype_model = pickle.load(f)
            if hasattr(subtype_model, 'feature_names_in_'):
                subtype_feature_columns = list(subtype_model.feature_names_in_)
        except:
            subtype_feature_columns = [
                'size', 'start_season', 'end_season', 'price_currency_id',
                'heating_type_id', 'building_age_id', 'city_id', 'county_id',
                'district_id', 'bedroom_count', 'living_room_count',
                'floor_no_id', 'tom', 'price', 'listing_type'
            ]

    except Exception as e:
        print(f"Error loading models: {e}")


# Загрузка данных
cities = []
districts = []
counties = []


def load_data():
    global cities, districts, counties

    try:
        # Загрузка городов
        city_df = pd.read_csv('df_city.csv')
        city_df = city_df.rename(columns={'id': 'city_id', 'data': 'city_name'})
        cities = city_df[['city_id', 'city_name']].to_dict('records')
    except:
        cities = []

    try:
        # Загрузка районов
        district_df = pd.read_csv('df_district.csv')
        district_df = district_df.rename(columns={'id': 'district_id', 'data': 'district_name'})
        districts = district_df[['district_id', 'district_name']].to_dict('records')
    except:
        districts = []

    try:
        # Загрузка округов
        county_df = pd.read_csv('df_county.csv')
        county_df = county_df.rename(columns={'id': 'county_id', 'data': 'county_name'})
        counties = county_df[['county_id', 'county_name']].to_dict('records')
    except:
        counties = []


# Инициализация при запуске
@app.on_event("startup")
async def startup_event():
    load_models()
    load_data()
    print("Models and data loaded successfully")


@app.get("/")
async def root():
    return {"message": "Real Estate Prediction API"}


@app.get("/health")
async def health_check():
    return {
        "price_model_loaded": price_model is not None,
        "subtype_model_loaded": subtype_model is not None,
        "cities_loaded": len(cities) > 0,
        "districts_loaded": len(districts) > 0,
        "counties_loaded": len(counties) > 0
    }


@app.get("/cities")
async def get_cities():
    return cities


@app.get("/districts")
async def get_districts():
    return districts


@app.get("/counties")
async def get_counties():
    return counties


@app.post("/predict/price")
async def predict_price(request: PricePredictionRequest):
    if price_model is None:
        return {"error": "Price model not loaded"}

    try:
        features = request.dict()
        features['price'] = 0  # Добавляем цену для совместимости

        input_data = pd.DataFrame([features])

        for col in price_feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[price_feature_columns]
        prediction = price_model.predict(input_data)[0]

        return {
            "predicted_price": float(prediction),
            "currency_id": request.price_currency_id,
            "listing_type": request.listing_type
        }

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}


@app.post("/predict/subtype")
async def predict_subtype(request: SubtypePredictionRequest):
    if subtype_model is None:
        return {"error": "Subtype model not loaded"}

    try:
        features = request.dict()
        features['sub_type_id'] = 1

        input_data = pd.DataFrame([features])

        for col in subtype_feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[subtype_feature_columns]

        prediction = subtype_model.predict(input_data)[0]
        probabilities = subtype_model.predict_proba(input_data)[0]
        confidence = max(probabilities)

        subtype_mapping = {
            1: "Квартира",
            2: "Частные дома",
            3: "Полное здание"
        }

        return {
            "predicted_subtype": subtype_mapping.get(prediction, "Неизвестно"),
            "confidence": float(confidence),
            "listing_type": request.listing_type
        }

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}


@app.post("/convert-currency")
async def convert_currency(request: CurrencyConversionRequest):
    currency_rates = {
        'TRY': 1.0,
        'USD': 0.024,
        'EUR': 0.020,
        'GBP': 0.017
    }

    try:
        amount_in_base = request.amount / currency_rates[request.from_currency]
        converted_amount = amount_in_base * currency_rates[request.to_currency]
        return {"converted_amount": converted_amount}
    except Exception as e:
        return {"error": f"Currency conversion error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
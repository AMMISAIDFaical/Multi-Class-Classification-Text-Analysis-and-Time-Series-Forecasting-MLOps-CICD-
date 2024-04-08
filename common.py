import pickle
from sklearn.model_selection import train_test_split
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from statsmodels.tsa.stattools import adfuller
from configparser import ConfigParser
import sqlite3
import os

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')
config = ConfigParser()
config.read(CONFIG_PATH)
DB_PATH = str(config.get("PATHS", "DB_PATH"))
MODEL_PATH = str(config.get("PATHS", "MODEL_PATH"))
RANDOM_STATE = int(config.get("ML", "RANDOM_STATE"))
# SQLite requires the absolute path
DB_PATH = os.path.join(ROOT_DIR, os.path.normpath(DB_PATH))

def retrieve_weather_data_as_pd(start_date="2020-01-01", end_date="2023-12-31"):
    print(f"getting data from the api")
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 46,
        "longitude": 2,
        "start_date": {start_date},
        "end_date": {end_date},
        "hourly": "temperature_2m",
        "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    ts_data = pd.DataFrame(data=hourly_data)
    return ts_data
def transform_data(ts_data):
    print("Transforming data to stationary data")
    return ts_data
def adf_test(ts_data):
    adftest = adfuller(ts_data['temperature_2m'], autolag='AIC', regression='ct')
    p_value = adftest[1]
    if p_value < 0.05:
        return True
def preprocess_data(ts_data):
    ts_data.dropna(inplace=True)
    print(ts_data.columns)
    ts_data["date"] = pd.to_datetime(ts_data["date"], format='%Y-%m-%d')
    ts_data = ts_data.set_index("date")
    ts_data = ts_data.resample('3h').mean()
    ts_data = ts_data[ts_data.index.year != 2019]
    if adf_test(ts_data):
        ts_data = transform_data(ts_data)
    for i in range(1, 2):
        ts_data[f"temperature_2m_lag_{i}"] = ts_data['temperature_2m'].shift(i)
    ts_data.dropna(inplace=True)
    return ts_data
def persist_data_in_db():
    print(f"Reading train data from the Weather API ..")
    ts_data = retrieve_weather_data_as_pd()
    prepro_ts_data = preprocess_data(ts_data)
    X = prepro_ts_data.drop('temperature_2m', axis=1)
    y = prepro_ts_data['temperature_2m']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    print("Training data min date : ")
    print(X_train.index.min())
    print("Training data max date : ")
    print(X_train.index.max())
    print("Test data min date : ")
    print(X_test.index.min())
    print("Test data max date : ")
    print(X_test.index.max())

    print(f'Xtrain shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')
    print(f'X_test shape : {X_test.shape}')
    print(f'y_test shape : {y_test.shape}')

    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    print(f"Saving train and test data to a database: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS X_train")
        cur.execute("DROP TABLE IF EXISTS X_test")
        cur.execute("DROP TABLE IF EXISTS y_train")
        cur.execute("DROP TABLE IF EXISTS y_test")
        X_train.to_sql(name='X_train', con=con, if_exists="replace")
        X_test.to_sql(name='X_test', con=con, if_exists="replace")
        y_train.to_sql(name='y_train', con=con, if_exists="replace")
        y_test.to_sql(name='y_test', con=con, if_exists="replace")
def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")
def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    print(model)
    return model
def persist_data_infos_db(model):
    print(f"Saving trained model data to the database: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as con:
        # Extracting coefficients and intercept
        coefficients = model.coef_
        intercept = model.intercept_
        # Converting coefficients to 1-dimensional array if needed
        if coefficients.ndim > 1:
            coefficients = coefficients.flatten()
        # Creating a pandas DataFrame
        data = {'Coefficient': coefficients}
        df = pd.DataFrame(data)
        # Adding intercept as a row
        df.loc['Intercept'] = intercept
        df.to_sql(name='model_infos'+str(model),if_exists='append', con=con)
def persist_preds_in_db(results_df):
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    print(f"Saving preds to a database: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as con:
        results_df.to_sql(name='results', con=con, if_exists='append')
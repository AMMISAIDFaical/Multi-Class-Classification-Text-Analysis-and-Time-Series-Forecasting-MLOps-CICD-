import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
import common
from common import persist_data_in_db

def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    X_train = pd.read_sql('SELECT * FROM X_train', con, parse_dates=['date'])
    y_train = pd.read_sql('SELECT * FROM y_train', con)
    con.close()
    return X_train, y_train

def fit_model(X_train, y_train):
    print(f"Fitting a model")
    lr = LinearRegression()
    X_train = X_train.set_index("date")
    y_train = y_train.set_index("date")
    print("Training data min date : ")
    print(X_train.index.min())
    print("Training data max date : ")
    print(X_train.index.max())
    print("Test data min date : ")
    lr.fit(X_train, y_train)
    return lr

if __name__ == "__main__":
    print("getting lastest data from the api and charge it to the db")
    persist_data_in_db()
    X_train, y_train = load_train_data(common.DB_PATH)
    model = fit_model(X_train, y_train)
    common.persist_model(model, common.MODEL_PATH)
    common.persist_data_infos_db(model)
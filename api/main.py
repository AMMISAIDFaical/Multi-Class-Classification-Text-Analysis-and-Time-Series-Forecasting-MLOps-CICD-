import json
import sqlite3
from datetime import datetime
from http.client import HTTPException

import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import common
from datetime import date

app = FastAPI(title="Time Series ML model for weather Temperatures prediction")
class Day(BaseModel):
    day_date: date

@app.on_event("startup") #This decorator ensures that the function loading the ml model is triggered when the Server starts.
def load_ml_pipeline():
    global model
    model = common.load_model("models/weather_timeseries.model")
def create_req_pred(req_pred_df):
    print(f"create Prediction and Request table in Db : {common.DB_PATH}")
    with sqlite3.connect(common.DB_PATH) as con:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS pred_req_history")
        req_pred_df.to_sql(name='pred_req_history', con=con, if_exists="replace")
    return req_pred_df

def pred_req_DbSave(results_df_filtered, req_pred_table):
    print(f"Saving Prediction and Request in Db: {common.DB_PATH}")
    # Creating a connection to the SQLite database
    with sqlite3.connect(common.DB_PATH) as con:
        cur = con.cursor()
        # Creating column list for insertion
        cols = ",".join([str(i) for i in results_df_filtered.columns.tolist()])
        for i, row in results_df_filtered.iterrows():
            # Convert datetime object to string format
            row['Days'] = row['Days'].strftime('%Y-%m-%d %H:%M:%S')
            # Building the SQL query with placeholders
            sql = f"INSERT INTO pred_req_history ({cols}) VALUES ({', '.join(['?' for _ in range(len(row))])})"
            print(sql)
            # Executing the SQL query with the values from the current row
            print(row)
            cur.execute(sql, tuple(row))
        # Committing changes to the database
        con.commit()

# Example usage:
# pred_req_DbSave(your_trip_prediction_df, "your_table_name")

# Returning the df as json keeping the datetime index in the iso format by setting orientation to table
def parse_csv(df):
    res = df.to_json(orient="table")
    parsed = json.loads(res)
    return parsed
def get_infrence_data(start_date="2023-12-31", end_date=None):
    ts_inferance_data = common.retrieve_weather_data_as_pd(start_date="2023-12-31", end_date=end_date)
    ts_inferance_data = common.preprocess_data(ts_inferance_data)
    X = ts_inferance_data.drop('temperature_2m', axis=1)
    return X

@app.post("/predict")
async def predict_temp(request: Day):
     # Convert the Pydantic model to a dictionary
    day_dict = request.dict()
    inf_date = day_dict['day_date']
    inf_date = str(inf_date.strftime("%Y-%m-%d"))
    X = get_infrence_data(start_date="2023-12-31", end_date=f"{inf_date}")
    # # # Make predictions
    prediction = model.predict(X)
    prediction = prediction.flatten()
    print(prediction)
    # Create a DataFrame with X,prediction
    results_df = pd.DataFrame({'Days': X.index,
                               'Predicted': prediction})
    # Filter rows for the date '2024-04-01'
    results_df_filtered = results_df[results_df['Days'].dt.date == pd.to_datetime('2024-04-01').date()]
     #creating table in the db to store request predection history
    req_pred_table = create_req_pred(results_df_filtered)
    #saving the filtred df in the history table
    pred_req_DbSave(results_df_filtered, req_pred_table)

    return parse_csv(results_df_filtered)


@app.get("/get_temp")
async def get_temp(day: str):
    con = sqlite3.connect(common.DB_PATH)  # Make sure common.DB_PATH is defined somewhere
    history_data = pd.read_sql('SELECT * FROM results WHERE date = ?', con, params=(day,))
    con.close()
    temperature_data = history_data['Predicted'].tolist()
    if history_data.empty:
        raise HTTPException(status_code=404, detail="Data not found for the specified day")

    return {"temperature": temperature_data}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True)
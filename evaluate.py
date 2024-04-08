import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

import common
def load_test_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    X_test= pd.read_sql('SELECT * FROM X_test', con, parse_dates=['date'])
    y_test = pd.read_sql('SELECT * FROM y_test', con)
    con.close()
    return X_test, y_test
def evaluate_model(model, X_test, y_test):
    # Make predictions using the model
    y_pred = model.predict(X_test)

    # Calculate the error
    error = y_test.values.flatten() - y_pred.flatten()

    # Create a DataFrame with the actual, predicted, and error values
    results_df = pd.DataFrame({
        'Actual': y_test.values.flatten(),
        'Predicted': y_pred.flatten(),
        'Error': error
    })
    results_df['date'] = (X_test.index)
    plt.figure(figsize=(10, 5))
    plt.plot(y_pred, "r", label="Prediction")
    plt.plot(y_test.values, label="Actual")
    plt.grid(True)
    plt.legend(loc="best")
    plt.title(
        f"Linear Regression\nMean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")
    plt.savefig('./visualization/monitoring/linear_reg_plot.png')
    return results_df

if __name__ == "__main__":
    X_test, y_test = load_test_data(common.DB_PATH)
    model = common.load_model(common.MODEL_PATH)
    X_test = X_test.set_index('date')
    y_test = y_test.set_index('date')

    results_df = evaluate_model(model, X_test, y_test)
    common.persist_preds_in_db(results_df)
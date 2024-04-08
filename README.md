# I. Historical Weather Time series forecasting

## II.Description of the project, architecture and data flows
In our project We utilized the Historical Weather API (Open-Meteo) to retrieve historical weather data. 
This API leverages reanalysis datasets sourced from diverse observations.
* For our data collection, we focused on obtaining a time series of temperature at 2m from the ground ("Temperature (2 m)")
for the City of France.
* We conducted experiments with different architectures such as Stats models and ML regression modlels Our primary objective was to develop a time series forecasting model using a combination of statistical methods and machine learning
regression techniques.

## III. Python Scripts
* after exprimentation we created three seperate python script files which encapsulate the ML pipeline 

  * train.py :
    * 1. `load_train_data`: This method loads training data from a SQLite database located at the specified path. 
          It returns the (`X_train`) and labels (`y_train`).
    * 2. `fit_model`: This method fits a linear regression model to the provided training data (`X_train` and `y_train`).
         It prints information about the training data, such as the minimum and maximum dates and returns the trained model.
    * 3. `persist_data_in_db`: This method is defined in the ***common.py script*** it retrieves the latest data from an API and stores it
          in the database processed and ready to train and test.
    * 4. `persist_model`: This method defined in ***common.py script*** it is responsible for persisting the trained model to the 
          path "/api/models"
    * 5. `persist_data_infos_db`: This method defined in ***common.py script*** it is responsible for persisting information 
          about the data, such as model parameters, to the database.

  * common.py :
    * 1. `retrieve_weather_data_as_pd(start_date="2020-01-01", end_date="2023-12-31")`: Retrieves weather data from an API (Open-Meteo) within 
            a specified date range. It prints information about the data, including coordinates, elevation, and timezone. 
            The data is then processed and returned as a Pandas DataFrame.
    * 2. `transform_data(ts_data)`: Transforms the retrieved weather data into stationary data NOT Implemented.
    * 3. `adf_test(ts_data)`: Conducts the Augmented Dickey-Fuller test to determine if the data is stationary. 
          If the p-value is less than 0.05, indicating stationarity, it returns True.
    * 4. `preprocess_data(ts_data)`: Preprocesses the retrieved weather data by dropping NaN values, converting date columns to datetime
         format, resampling the data, checking for stationarity, and creating lag features. 
         The preprocessed data is returned as a Pandas DataFrame.
    * 5. `persist_data_in_db()`: Retrieves weather data from the API, preprocesses it, splits it into training and testing sets, and saves 
          the sets into a SQLite database.
    * 6. `persist_model(model, path)`: Persists a trained model to a specified path using pickle serialization.
    * 7. `load_model(path)`: Loads a trained model from a specified path using pickle deserialization.
    * 8. `persist_data_infos_db(model)`: Saves information about the trained model, such as coefficients and intercept, to the SQLite database.
    * 9. `persist_preds_in_db(results_df)`: Saves predictions to the SQLite database.
      
  * evaluate.py :
    * 1. `load_test_data(path)`: Reads test data from a SQLite database located at the specified path. 
         It retrieves features (`X_test`) and labels (`y_test`) from the database and returns them.
    * 2. `evaluate_model(model, X_test, y_test)`: Evaluates the performance of a given model using the test data.It makes predictions using the model,
         calculates the error between the actual and predicted values, and visualizes the results using a line plot. The mean absolute percentage error (MAPE) is also calculated and included in the plot title. 
         The function returns a DataFrame containing actual, predicted, and error values.
    * 3. `persist_preds_in_db(results_df)`: Saves predictions to a SQLite database. The results DataFrame, containing actual, predicted, and error values, is saved to the database.

## Serving Model using FAST API
  * main.py : 
     * 1. `load_ml_pipeline()`: Loads the machine learning model during server startup. The model is loaded using a function from the `common` module.
   
     * 2. `create_req_pred(req_pred_df)`: Creates the Prediction and Request table in the database (`common.DB_PATH`) and replaces it if it already exists. It takes a DataFrame (`req_pred_df`) containing prediction and request data as input and returns the same DataFrame.
     
     * 3. `pred_req_DbSave(results_df_filtered, req_pred_table)`: Saves prediction and request data to the database. It iterates over rows of a DataFrame (`results_df_filtered`) and inserts them into the `pred_req_history` table. The table name is passed as an argument (`req_pred_table`).
     
     * 4. `parse_csv(df)`: Converts a DataFrame (`df`) to JSON format with datetime index in ISO format. It returns the parsed JSON.
     
     * 5. `get_infrence_data(start_date="2023-12-31", end_date=None)`: Retrieves inference data from the weather API and preprocesses it. It returns the preprocessed data excluding the 'temperature_2m' column.
     
     * 6. `predict_temp(request: Day)`: Handles POST requests to predict temperature. It receives a `Day` object as input containing the date for which temperature prediction is requested. It makes predictions using the loaded machine learning model and returns the predictions for the specified date in JSON format.
     
     * 7. `get_temp(day: str)`: Handles GET requests to retrieve temperature data for a specific day. It queries the database for temperature predictions made for the specified day and returns the data in JSON format.

## III. Dockerizing the Server
  word on the ***Server*** : 
  * As our Fast API has been built it runs locally and it has two method : POST http predict (returns temperatures of the day inserted) and get_temp gets the preds for giving day date if its available in our database, the Uvicorn Server can be use the API to serve the prediction requests. 
 * for dockerizing our api we created DOCKERFILE with main components :
    # Dockerfile 
      ### Base Image
          - FROM python:latest**: Specifies the base image for your Docker container, in this case, it's the latest version of Python available from Docker Hub.
      ### System Packages
          - RUN apt-get update && apt-get upgrade -y && \ apt-get install -y --no-install-recommends bash git openssh-client**: updates image
      ### Working Directory
          - WORKDIR ./app**: Sets the working directory inside the container to `/app`.
      ### Application Setup
          - COPY requirements.txt .**: Copies the `requirements.txt` file from the Docker build context to the `/app` directory inside the container.
             * requirements has to containe :
                - is project requires specific Python packages for data analysis and web development, including pandas, numpy, fastjsonschema, uvicorn, fastapi, pydantic, scikit-learn, openmeteo-requests,   
                  requests-cache, retry-requests, statsmodels, and configparser, with version constraints ensuring compatibility and best practices for installation and version management.
          - RUN pip install -r requirements.txt**: Installs Python dependencies listed in `requirements.txt` using pip.
     ### Application Files
          - COPY ./api /app**: Copies the contents of the `api` directory from the Docker build context to the `/app` directory inside the container.
          - COPY ./common.py .**: Copies the `common.py` file from the Docker build context to the current working directory inside the container.
          - COPY ./config.ini .**: Copies the `config.ini` file from the Docker build context to the current working directory inside the container.
     ## Command
     - **CMD ["python", "main.py"]**: Specifies the command that will be executed when the container starts. In this case, it runs the `main.py` script using Python.
## IV. Pushing the work to personal github repo

## V. Writing workflow yml files build 
   ### Build Workflow Documentation CI/CD
Workflow Name
- name: Build and Run Docker Image**: Describes the purpose of the workflow, which is to build and run a Docker image.

Triggers
- **on**: Specifies the triggers for the workflow.
- **workflow_dispatch**:  manual triggering of the workflow.

Jobs
- **build**: Defines the job to build the Docker image.
- **permissions**: Specifies permissions required for the job.
- **packages: write**: Grants write permissions for packages.
- **name**: Describes the job.
- **runs-on**: Specifies the operating system for the job (ubuntu-latest).
      
Steps
- **Checkout code**: Checks out the repository code using the `actions/checkout` action.
- **Build Docker image**: Builds the Docker image using the `docker build` command, tagging it as `ts-project`.
- **Push Docker image**: Pushes the Docker image to GitHub Packages.
- Logs into Docker registry using GitHub token stored in secrets.
- Tags the Docker image with the appropriate repository URL.
- Pushes the tagged Docker image to GitHub Packages registry.

 # Historical Weather Time Series Forecasting

## High-level Description
This project aims to forecast historical weather time series data using machine learning regression techniques. It utilizes the Historical Weather API (Open-Meteo) to retrieve temperature data at 2m from the ground for the City of France. The primary objective is to develop accurate time series forecasting models through experimentation with different architectures such as statistical models and machine learning regression models.

## Data Flow & Architecture
The data flow involves retrieving weather data from the Historical Weather API, preprocessing it, training machine learning models, evaluating model performance, and serving predictions through a FastAPI server. The architecture encompasses Python scripts for data retrieval, preprocessing, model training, evaluation, and serving, as well as Docker for containerization of the FastAPI server.

## Main Technologies Used and for Which Purpose
- Python: Main programming language for scripting, data manipulation, and model development.
- Pandas, NumPy: Data manipulation and preprocessing.
- Statsmodels, scikit-learn: Statistical models and machine learning regression techniques.
- FastAPI: Web framework for serving machine learning models.
- Docker: Containerization of the FastAPI server for deployment.

## Running Locally
To run the project locally, follow these steps:

### Install Dependencies
1. Ensure you have Python installed on your system.
2. Clone the repository.
3. Navigate to the project directory.
4. Install dependencies in requirement.txt make sure you have the right python venv that contains the sqlite dependencie
5. listing the dependencies in requirement.txt:
pandas~=2.2.1
numpy~=1.26.4
fastjsonschema==2.19.0
uvicorn~=0.24.0.post1
fastapi~=0.104.1
pydantic~=2.5.2
scikit-learn>=0.24.0
openmeteo-requests~=1.2.0
requests-cache
retry-requests
statsmodels
configparser


### Run
1. Execute the train.py script to retrive data using the common class save it to db
   and train model and saves it as pickel in api/model/ folder (make sure you adjust the paths  
   for local run
2. Execute evaluate.py for model evaluation , new tables will be added to the db and to 
   visualization/monitoring linear_reg_plot.png.
3. Execute the main.py the app will run on the port 8000 by adding /docs to the giving http url
   you can see swagger ui and test api endpoints (predict day 2024-04-01) always try for 5 days    in past because api cant provide data till the current day you are in

### Build docker image 
   1. on the terminal run docker build -t ts-prod .
   2. the if build of the image is perfectly built run : docker run -d -p 8000:8000 ts-prod
   3. you should get the app runing on same local port 8000 but in docker container 

### Test
To test the project locally (docker container), you can manually make HTTP requests to the FastAPI server endpoints using swagger ui.

## CI/CD Steps

### Step 1: Build and Run Docker Image

- **Description**: This step builds a Docker image named `ts-project` from the repository code and pushes it to GitHub Packages.
- **Outputs**:
  - If successful, the Docker image will be pushed to GitHub Packages under the repository `you_username/time_series_mlops/ts-project`.

### Step 2: Run Automatic Tests

- **Description**: This step pulls the Docker image from GitHub Packages, runs it as a container, executes automatic tests on the API, and prints the container logs before and after the test.
- **Outputs**:
  - Container logs before curl request.
  - Result of the API test request (e.g., response code, response body).
  - Container logs after curl request.




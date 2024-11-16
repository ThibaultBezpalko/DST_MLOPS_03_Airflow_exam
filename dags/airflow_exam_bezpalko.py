from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.sensors.base import BaseSensorOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, get_current_context
from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.models.xcom import LazyXComAccess
from airflow.utils.decorators import apply_defaults

# API libs
import requests
import json

# Global libs
import sys
import datetime
from pathlib import Path
import os
import glob

# Data preparation libs
import numpy as np
import pandas as pd

# Model libs
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


# API key to call the OpenWeatherMap API
API_key = 'dd48455b8b6454dfa07d39a4f69de373'

# List of cities to collect the data
cities = ['paris', 'london', 'washington']
Variable.set(key="cities", value=json.dumps(cities))

# Minimal number of files for dashboard dataset
dashboard_files_number = 20

# Instantiate the models
models_dict = {
    'Linear Regression': LinearRegression(), 
    'Decision Tree Regression': DecisionTreeRegressor(), 
    'Random Forest Regression': RandomForestRegressor()
}

# Paths
Variable.set(key="model_path", value='/app/model/best_model.pickle')


# Task 01: get data from OpenWeatherMap API for each city minutely
@task(task_id="Calling_OpenWeatherMap_API", task_concurrency=1)
def owm_request_get():    
    # json filename:
    dt0 = datetime.datetime.now()
    dt = dt0.strftime("%Y-%m-%d %H:%M")
    filename = dt + '.json'

    # Check if the file exists
    try:
        with open(f'/app/raw_files/{filename}', 'r') as file:
            print(f"The file '{filename}' already exists")
        # To avoid double JSON array object when starting the DAG
        pass
    # If not, collect data
    except FileNotFoundError:
        print(f"The file '{filename}' doesn't exist. Collecting data")

        # Loading list of cities from Airflow variable
        airflow_cities_list = Variable.get(key="cities", deserialize_json=True)
        # airflow_cities_list = json.loads(airflow_cities)
        print(f"List of cities to get measures: {airflow_cities_list}")

        # Instantiate the data list collecting data from API
        data = []

        # Collecting data for each city
        for city in airflow_cities_list:
            dt_i = datetime.datetime.now()
            print(f'Call to https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_key} at {dt_i}')
            r = requests.get(
                url='https://api.openweathermap.org/data/2.5/weather', 
                params={"q": city, "appid": API_key}
            )
            if r.status_code == 200:
                # Append each JSON response to the data object
                data.append(r.json())  
            else:
                print(f"Error with status code: {r.status_code}")
        
            dt_f = datetime.datetime.now()
            # Current duration of API call
            print(f"Call time : {dt_f - dt_i} s")

        # Write the combined JSON array to a file
        # If some correct data is present in the data object, then create a file
        if data != []:
            with open(f'/app/raw_files/{filename}', 'a') as file:
                json.dump(data, file, indent=4)  # Writes the list as a JSON array
                print(f"Data successfully written to {filename}!") 
        else:
            print("No data to create a file!")


# Checking the number of json files before starting the rest of process
@task(task_id="Check_Number_JSON")
def check_files(directory, pattern, required_count):
    files = glob.glob(os.path.join(directory, pattern))
    if len(files) < required_count:
        raise ValueError(f"Not enough files found! Expected at least {required_count}, found {len(files)}")
    print(f"Found {len(files)} files matching the pattern '{pattern}'.")


# Task 02 and 03 : prepare the csv files used for dashboard and model training
def transform_data_into_csv(n_files=None, filename='fulldata.csv'):
    parent_folder = '/app/raw_files'

    # Sort the files chronogically
    files = sorted(os.listdir(parent_folder), reverse=True)
    if n_files:
        files = files[:n_files]
    print(files)

    # Instantiate the list of data
    dfs = []

    # Retrieve the targeted meteo data
    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)
            for data_city in data_temp:
                dfs.append(
                    {
                        'temperature': data_city['main']['temp'],
                        'city': data_city['name'],
                        'pression': data_city['main']['pressure'],
                        'date': f.split('.')[0]
                    }
                )

    # Conversion into a DataFrame before creating the csv file
    df = pd.DataFrame(dfs)

    print('\n', df.head(100))

    # Create a csv file fill with data
    df.to_csv(os.path.join('/app/clean_data', filename), index=False)


# Task 04: cross-validation for the selected model fed with data
def compute_model_score(model, X, y):
    # computing cross val
    cross_validation = cross_val_score(
        model,
        X,
        y,
        cv=2,
        scoring='neg_mean_squared_error')

    # Mean score for the cross validated runs
    model_score = cross_validation.mean()

    return model_score


# Prepare features and target before fitting model
def prepare_data(path_to_data='/app/clean_data/fulldata.csv'):
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for c in df['city'].unique():
        df_temp = df[df['city'] == c].copy()

        # creating target
        df_temp.loc[:, 'target'] = df_temp.loc[:, 'temperature'].shift(1)

        # creating features
        for i in range(1, 5):
            df_temp.loc[:, f'temp_m-{i}'] = df_temp.loc[:, 'temperature'].shift(-i)

        # deleting null values
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # concatenating datasets
    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )

    # deleting date variable
    df_final = df_final.drop(['date'], axis=1)

    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)
    print(df_final.head())

    features = df_final.drop(['target'], axis=1)
    target = df_final['target']

    return features, target

# Train and evaluate the model
def train_evaluate_model(models_dict, model_name, task_instance):
    # Combining json files to prepare the data
    X, y = prepare_data('/app/clean_data/fulldata.csv')

    # Cross-validation of the model
    score = compute_model_score(models_dict[model_name], X, y)
    neg_mean_squared_error = score.mean()
    print(f"{model_name} Score: {neg_mean_squared_error}")


    # Create an Airflow Xcom value to be used in model selection
    task_instance.xcom_push(
        key=f"{model_name} Score",
        value={
            model_name: neg_mean_squared_error
        }
    )


# Task 05: choose the best model
# Retrain the best model and save it
def train_and_save_model(best_model):
    # Get the features
    X, y = prepare_data('/app/clean_data/fulldata.csv')

    # Training the model
    model = models_dict[best_model]
    model.fit(X, y)

    # Saving model
    path_to_model = Variable.get(key="model_path")
    print(str(best_model), 'saved at ', path_to_model)
    dump(model, path_to_model)


# Compare the scores of the trained model
@task(task_id="Choosing_Model")
def choose_model(models_dict, task_instance):
    lr_score = task_instance.xcom_pull(
        key="Linear Regression Score",
        task_ids='Evaluate_Models.Linear_Regression'
    )
    dtr_score = task_instance.xcom_pull(
        key="Decision Tree Regression Score",
        task_ids='Evaluate_Models.Decision_Tree_Regression'
    )
    rfr_score = task_instance.xcom_pull(
        key="Random Forest Regression Score",
        task_ids='Evaluate_Models.Random_Forest_Regression'
    )
    
    # Instantiate the scores dict
    neg_mse_scores = {**lr_score, **dtr_score, **rfr_score}
    print(type(neg_mse_scores))
    print(neg_mse_scores) 

    # Summary of results
    for key, value in neg_mse_scores.items():
        print(f"{key}: Mean Negative MSE = {value}")

    # Selection of best model with the highest neg_mse (closest to 0)
    best_model = max(neg_mse_scores, key=neg_mse_scores.get)
    print(f"The best model is {best_model} with a score of {neg_mse_scores[best_model]}")

    # Retrain and save the best model
    train_and_save_model(best_model)



# Define the DAG using the traditional method
with DAG(
    dag_id='airflow_exam_dag',
    tags=['exam', 'datascientest'],
    doc_md='''
    DataScientest Sep 2024 - MLOPS
    Examen Airflow Thibault BEZPALKO

    This DAG is composed of:
    - minutely API call to the OpenWeatherMap to get temperature and pressure variables for Paris, London, Washington
    - prepare the datasets for dashboard feeding and for model training
    - train 3 models
    - choose and save the best model 
    ''',
    schedule_interval=datetime.timedelta(seconds=60),
    start_date=days_ago(0),
    catchup=False,
    max_active_runs=1,  # Ensure only one DAG run at a time
) as dag:

    # Calling OpenWeatherMap API
    t_request = owm_request_get()

    # Wait for enough files before next process
    t_wait = check_files(
        directory="/app/raw_files/",
        pattern="*.json",
        required_count=dashboard_files_number,
    )
    
    with TaskGroup("Prepare_Datasets", tooltip="Preparing datasets tasks group") as t_prepare_data:
        # Preparing dataset for dashboard
        t_dataset_dashboard = PythonOperator(
            task_id='Preparing_Dataset_Dashboard',
            dag=dag,
            python_callable=transform_data_into_csv,
            op_kwargs={'n_files':dashboard_files_number, 'filename':'data.csv'},
        )

        # Preparing dataset for training model
        t_dataset_model = PythonOperator(
            task_id='Preparing_Dataset_Model',
            dag=dag,
            python_callable=transform_data_into_csv,
        )

    # Training tasks group
    with TaskGroup("Evaluate_Models", tooltip="Training models tasks group") as t_train_models:
        lr = PythonOperator(
            task_id='Linear_Regression',
            dag=dag,
            python_callable=train_evaluate_model,
            op_kwargs={'models_dict': models_dict, 'model_name': 'Linear Regression'},
        )

        dtr = PythonOperator(
            task_id='Decision_Tree_Regression',
            dag=dag,
            python_callable=train_evaluate_model,
            op_kwargs={'models_dict': models_dict, 'model_name': 'Decision Tree Regression'},
        )

        rfr = PythonOperator(
            task_id='Random_Forest_Regression',
            dag=dag,
            python_callable=train_evaluate_model,
            op_kwargs={'models_dict': models_dict, 'model_name': 'Random Forest Regression'},
        )

    t_choose_model = choose_model(models_dict=models_dict)

    # Dependencies
    t_request >> t_wait
    t_wait >> t_prepare_data
    # t_wait >> [t_dataset_dashboard, t_dataset_model]
    t_dataset_model >> t_train_models
    t_train_models >> t_choose_model

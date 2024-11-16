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
Variable.set(key="cities", value=cities)
my_variable_value = Variable.get(key="cities")

# Number of files for dashboard dataset
dashboard_files_number = 20

# Instantiate the models
models_dict = {
    'Linear Regression': LinearRegression(), 
    'Decision Tree Regression': DecisionTreeRegressor(), 
    'Random Forest Regression': RandomForestRegressor()
}


@task(task_id="Calling_OpenWeatherMap_API", task_concurrency=1)
def owm_request_get():
    expected_code = 200
    
    # json file:
    dt0 = datetime.datetime.now()
    dt = dt0.strftime("%Y-%m-%d %H:%M")
    filename = dt + '.json'

    # Check if the file exists and read the previous data if available
    try:
        with open(f'/app/raw_files/{filename}', 'r') as file:
            print(f"The file '{filename}' already exists")
    except FileNotFoundError:
        print(f"The file '{filename}' doesn't exist")

    data = []

    for city in cities:
        dt_i = datetime.datetime.now()
        print(f'Call to https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_key} at {dt_i}')
        r = requests.get(
            url='https://api.openweathermap.org/data/2.5/weather', 
            params={"q": city, "appid": API_key}
        )
        if r.status_code == 200:
            data.append(r.json())  # Append each JSON response to the list
        else:
            print(f"Error with status code: {r.status_code}")
       
        dt_f = datetime.datetime.now()
        # Current duration of API call
        print(f"Call time : {dt_f - dt_i} s")

    # Write the combined JSON array to a file
    with open(f'/app/raw_files/{filename}', 'w') as file:
        json.dump(data, file, indent=4)  # Writes the list as a JSON array
        print(f"Data successfully written to {filename}") 


@task(task_id="Check_Number_JSON")
def check_files(directory, pattern, required_count):
    files = glob.glob(os.path.join(directory, pattern))
    if len(files) < required_count:
        raise ValueError(f"Not enough files found! Expected at least {required_count}, found {len(files)}")
    print(f"Found {len(files)} files matching the pattern '{pattern}'.")


def transform_data_into_csv(n_files=None, filename='fulldata.csv'):
    parent_folder = '/app/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    if n_files:
        files = files[:n_files]
    print(files)

    dfs = []

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

    df = pd.DataFrame(dfs)

    print('\n', df.head(100))

    df.to_csv(os.path.join('/app/clean_data', filename), index=False)


def compute_model_score(model, X, y):
    # computing cross val
    cross_validation = cross_val_score(
        model,
        X,
        y,
        cv=2,
        scoring='neg_mean_squared_error')

    model_score = cross_validation.mean()

    return model_score


def train_and_save_model(model, path_to_model='/app/clean_data/best_model.pickle'):
    # Combining json files to prepare the data
    X, y = prepare_data('/app/clean_data/fulldata.csv')

    # training the model
    model.fit(X, y)

    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)


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
            df_temp.loc[:, 'temp_m-{}'.format(i)] = df_temp.loc[:, 'temperature'].shift(-i)

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

    features = df_final.drop(['target'], axis=1)
    target = df_final['target']

    return features, target


def train_evaluate_model(models_dict, model_name, task_instance):
    # Combining json files to prepare the data
    X, y = prepare_data('/app/clean_data/fulldata.csv')

    # Cross-validation of the model
    score = compute_model_score(models_dict[model_name], X, y)
    neg_mean_squared_error = score.mean()

    task_instance.xcom_push(
        key="my_xcom_value",
        value={
            model_name: neg_mean_squared_error
        }
    )


@task(task_id="Choosing_Model")
def choose_model(models_dict, task_instance):
    lr_score = task_instance.xcom_pull(
        key="my_xcom_value",
        task_ids='Evaluate_Models.Linear_Regression'
    )
    dtr_score = task_instance.xcom_pull(
        key="my_xcom_value",
        task_ids='Evaluate_Models.Decision_Tree_Regression'
    )
    rfr_score = task_instance.xcom_pull(
        key="my_xcom_value",
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
    train_and_save_model(
                models_dict[best_model],
                '/app/clean_data/best_model.pickle'
            )


# Define the DAG using the traditional method
with DAG(
    dag_id='airflow_exam_dag',
    tags=['exam', 'datascientest'],
    doc_md='''Documentation dag''',
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

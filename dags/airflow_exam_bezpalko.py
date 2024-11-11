from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.models import Variable
import random
import requests
import sys
import json
import datetime
from pathlib import Path
import os
import datetime

# List of cities to collect the data
cities = ['paris', 'london', 'washington']
Variable.set(key="cities", value=cities)
my_variable_value = Variable.get(key="cities")

# API key to call the OpenWeatherMap API
API_key = 'dd48455b8b6454dfa07d39a4f69de373'

@task
def owm_request_get():
    expected_code = 200
    
    # json file:
    dt0 = datetime.datetime.now()
    dt = dt0.strftime("%Y-%m-%d %H:%M")
    filename = dt + '.json'

    data = []

    for city in cities:
        print(f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_key}')
        r = requests.get(
            url='https://api.openweathermap.org/data/2.5/weather', 
            params={"q": city, "appid": API_key}
        )
        if r.status_code == 200:
            data.append(r.json())  # Append each JSON response to the list
        else:
            print(f"Error with status code: {r.status_code}")

        print("Data successfully written to combined_data.json")        
        dt1 = datetime.datetime.now()
        # storing the current time in the variable
        print(f"Call time : {dt1 - dt0} s")

    # Write the combined JSON array to a file
    with open(f'/app/raw_files/{filename}', 'a') as file:
        json.dump(data, file, indent=4)  # Writes the list as a JSON array
        file.write('\n')



@task
def transform_data_into_csv(_, n_files=None, filename='data.csv'):
    parent_folder = '/app/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    if n_files:
        files = files[:n_files]

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

    print('\n', df.head(10))

    df.to_csv(os.path.join('/app/clean_data', filename), index=False)


@dag(
    dag_id='airflow_exam_dag',
    tags=['exam', 'datascientest'],
    doc_md='''Documentation dag''',
    schedule_interval=datetime.timedelta(seconds=60),
    start_date=days_ago(0),
    catchup=False
)
def dag():
    task_01 = owm_request_get()
    #task_02 = transform_data_into_csv(task_01.output, 5, 'data.csv', )
    #task_03 = transform_data_into_csv(task_01.output, None, 'fulldata.csv')

dag = dag()


'''
def function_with_return(task_instance):
    task_instance.xcom_push(
        key="my_xcom_value",
        value=random.uniform(a=0, b=1)
    )

def read_data_from_xcom(task_instance):
    print(
        task_instance.xcom_pull(
            key="my_xcom_value",
            task_ids=['python_task']
        )
    )

with DAG(
    dag_id='simple_xcom_dag',
    schedule_interval=None,
    start_date=days_ago(0)
) as my_dag:

    my_task = PythonOperator(
        task_id='python_task',
        python_callable=function_with_return
    )

    my_task2 = PythonOperator(
        task_id="read_Xcom_value",
        python_callable=read_data_from_xcom
    )

    my_task >> my_task2


with DAG(
    dag_id='sensor_dag',
    schedule_interval=None,
    tags=['tutorial', 'datascientest'],
    start_date=days_ago(0)
) as dag:

    my_sensor = FileSensor(
        task_id="check_file",
        fs_conn_id="my_filesystem_connection",
        filepath="/tmp/my_file.txt",
        poke_interval=30,
        timeout=5 * 30,
        mode='reschedule'
    )

    my_task = BashOperator(
        task_id="print_file_content",
        bash_command="cat /tmp/my_file.txt",
    )




@task
def function_with_return_and_push():
    task_instance = get_current_context()['task_instance']
    value = random.uniform(a=0, b=1)
    task_instance.xcom_push(key="my_xcom_value", value=value)
    return value

@task
def read_data_from_xcom(my_xcom_value):
    print(my_xcom_value)
'''
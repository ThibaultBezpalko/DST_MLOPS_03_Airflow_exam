import os
import json
import numpy as np
import pandas as pd

def transform_data_into_csv(n_files=None, filename='fulldata.csv'):
    parent_folder = './raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    print(files)
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

    df.to_csv(os.path.join('./clean_data', filename), index=False)

if __name__ == '__main__':
    transform_data_into_csv(20, 'data.csv')
    transform_data_into_csv()
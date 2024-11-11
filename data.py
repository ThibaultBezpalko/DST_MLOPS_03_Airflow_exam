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

# API key to call the OpenWeatherMap API
API_key = 'dd48455b8b6454dfa07d39a4f69de373'

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
    with open(f'./raw_files/{filename}', 'a') as file:
        json.dump(data, file, indent=4)  # Writes the list as a JSON array


if __name__ == '__main__':
    owm_request_get()


import pandas as pd


data ={
    'id': [3585, 5815, 2447, 3571, 9431, 2314, 9817, 7522, 9701, 7326],
    'name': ['OZTVQ', 'VUIHW', 'LNOVK', 'DCGHO', 'UPHET', 'GBMJC', 'MZAOB', 'NISPH', 'GAHPN', 'UBJFB'],
    'age': [30, 43, 52, 27, 51, 22, 31, 44, 50, 36],
    'salary': [98209.0, 86842.82, 70421.4, 49865.81, 30288.9, 96610.23, 36677.24, 74866.86, 70814.43, 85702.4],
    'city': ['New York', 'Houston', 'Miami', 'New York', 'New York', 'New York', 'Houston', 'Miami', 'Houston', 'Miami']
}

df = pd.DataFrame(data)

df.head()
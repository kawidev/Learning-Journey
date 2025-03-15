import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Tworzenie DataFrame
data = {
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, np.nan, 22, 35],
    'Gender': ['Female', 'Male', 'Male', np.nan, 'Female'],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'New York'],
    'Salary': [50000, 60000, 70000, np.nan, 80000],
    'Married': [True, False, True, False, np.nan],
    'Education': ['Bachelor', 'Master', 'PhD', 'Bachelor', np.nan]
}

df = pd.DataFrame(data)
#print(df)

# Podziel dane na zbiór treningowy (X_train) i walidacyjny (X_valid):
# Na przykład, X_train może zawierać pierwsze 3 wiersze, a X_valid ostatnie 2 wiersze.

X_train, X_valid = train_test_split(df, test_size=0.4, random_state=0)
#print(X_train)

X_train = X_train.drop(['ID'], axis=1)
X_valid = X_valid.drop(['ID'], axis=1)

X_train = X_train.set_index('Name')
X_valid = X_valid.set_index('Name')

object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
#print(good_label_cols)

# Utwórz obiekt OrdinalEncoder
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Wytrenuj OrdinalEncoder na danych treningowych
X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

#print(X_train)
#print(X_valid)

object_unique = list(map(lambda x: len(X_train[x].unique()), object_cols))
print(object_unique)
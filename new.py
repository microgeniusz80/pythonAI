import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import plot_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
print(insurance)

insurance_one_hot = pd.get_dummies(insurance)

insurance_one_hot['region_northeast'] = insurance_one_hot['region_northeast'] .astype(int)
insurance_one_hot['region_northwest'] = insurance_one_hot['region_northwest'] .astype(int)
insurance_one_hot['region_southeast'] = insurance_one_hot['region_southeast'] .astype(int)
insurance_one_hot['region_southwest'] = insurance_one_hot['region_southwest'] .astype(int)
insurance_one_hot['sex_female'] = insurance_one_hot['sex_female'] .astype(int)
insurance_one_hot['sex_male'] = insurance_one_hot['sex_male'] .astype(int)
insurance_one_hot['smoker_no'] = insurance_one_hot['smoker_no'] .astype(int)
insurance_one_hot['smoker_yes'] = insurance_one_hot['smoker_yes'] .astype(int)

ct = make_column_transformer(
    (
        MinMaxScaler(),
        [
            'age',
            'bmi',
            'children'
        ]
    ),
    (
        OneHotEncoder(handle_unknown='ignore'),
        [
            'sex',
            'smoker',
            'region'
        ]
    )
)

X = insurance_one_hot.drop('charges', axis=1)
y = insurance_one_hot['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tf.random.set_seed(42)

insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae']
)

history = insurance_model.fit(X_train, y_train, epochs=250)

print(insurance_model.evaluate(X_test, y_test))

print(y_train)

pd.DataFrame(history.history).plot()
plt.show()


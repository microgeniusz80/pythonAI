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

X = insurance.drop('charges', axis=1)
y = insurance['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ct.fit(X_train)

X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

print(X_train_normal)

tf.random.set_seed(42)

insurance_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1),
    ]
)

insurance_model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae']
)

insurance_model.fit(X_train_normal, y_train, epochs=1000)

print('evaluate: ', insurance_model.evaluate(X_test_normal, y_test))
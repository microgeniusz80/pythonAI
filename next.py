import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import plot_model
import pandas as pd



# X = tf.range(-100, 100, 4)
# y = X + 10

# X_train = X[:40]
# y_train = y[:40]

# X_test = X[40:]
# y_test = y[40:]

# # print(len(X_train), len(y_train))
# # print(len(X_test), len(y_test))

# # plt.figure(figsize=(10, 7))
# # plt.scatter(X_train, y_train, c='b', label='training data')
# # plt.scatter(X_test, y_test, c='g', label='testing data')
# # plt.legend()
# # plt.show()

# tf.random.set_seed(42)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(100, input_shape=[1], name='input_layer'),
#     tf.keras.layers.Dense(1, name='output_layer')
# ])

# model.compile(
#     loss=tf.keras.losses.mae,
#     optimizer=tf.keras.optimizers.SGD(),
#     metrics=['mae']
# )

# model.fit(X_train, y_train, epochs=100, verbose=0)

# y_pred = model.predict(X_test)

# plt.figure(figsize=(10, 7))
# plt.scatter(X_train, y_train, c='b', label='training data')
# plt.scatter(X_test, y_test, c='g', label='testing data')
# plt.scatter(X_test, y_pred, c='r', label='testing data')
# plt.legend()
# plt.show()

# model.save('model_test')

# print(model.summary())

# print(keras.utils.plot_model(model, show_layer_names=True, show_shapes=True, to_file='model.png'))

insurance = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
print(insurance)
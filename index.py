import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

X = tf.cast(tf.constant(X), dtype = tf.float32)
y = tf.cast(tf.constant(y), dtype = tf.float32)

# plt.scatter(X, y)
# plt.show()

tf.random.set_seed (42)

model = tf.keras.Sequential (
    [
        tf.keras.layers.Dense(50, activation=None),
        tf.keras.layers.Dense(1)
    ]
)

model.compile (
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    metrics = ['mae']
)

model.fit (
    tf.expand_dims(X, axis = -1), 
    y, 
    epochs = 100
)

y_pred = model.predict(
    [17.0]
)

print(y_pred)
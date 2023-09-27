from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

tf.random.set_seed(42)
n_samples = 1000

X, y = make_circles(
    n_samples,
    noise=0.03,
    random_state=42
)

# print(X.shape, y.shape)
# print(X[:10])
# print(y[:10])

# plt.scatter(
#     X[:, 0],
#     X[:, 1],
#     c=y,
#     cmap=plt.cm.RdYlBu
# )

# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train), len(y_train), len(X_test), len(y_test))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# tf.keras.layers.Dense(100, input_shape=(None, 1))
# model_3.fit(tf.expand_dims(X_reg_train, axis=-1),

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# model.fit(X_train, y_train, epochs=300)

# model.fit(X_train, y_train, epochs=100)
# print(model.evaluate(X_test, y_test))


# print(model.evaluate(X_test, y_test))
# print(model.predict(X_train))

def plot_decision_boundary(model, X, y):
    """Plots the decision boundary created by a model predicting on X.

    Args:
        model (tensorflow model): sample model trained
        X (tensors): features
        y (tensors): labels
    """
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    x_in = np.c_[
        xx.ravel(),
        yy.ravel()
    ]
    
    y_pred = model.predict(x_in)
    
    if len(y_pred[0]) > 1:
        print('doing multi class classification')
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print('doing binary classification')
        y_pred = np.round(y_pred).reshape(xx.shape)
    
    plt.contourf(
        xx, 
        yy, 
        y_pred, 
        cmap=plt.cm.RdYlBu,
        alpha=0.7
    )
    
    plt.scatter(
        X[:,0], 
        X[:,1],
        c=y,
        s=40,
        cmap=plt.cm.RdYlBu
    )
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.show()
    
    
# plot_decision_boundary(model = model, X=X, y=y)


X_regression = tf.range(0, 1000, 5)
y_regression = tf.range(100, 1100, 5)

X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]

y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]

# model.fit(X_reg_train, y_reg_train, epochs=100)

model_reg = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1),
    ]
)

model_reg.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae']
)

# model_reg.fit(tf.expand_dims(X_reg_train, axis=-1), y_reg_train, epochs=100)

# y_reg_preds = model_reg.predict(X_reg_test)

# plt.figure(figsize=(10, 7))

# plt.scatter(
#     X_reg_train, 
#     y_reg_train,
#     c= 'b',
#     label='training data' 
# )

# plt.scatter(
#     X_reg_test, 
#     y_reg_test,
#     c= 'g',
#     label='testing data' 
# )

# plt.scatter(
#     X_reg_test, 
#     y_reg_preds,
#     c= 'r',
#     label='prediction data' 
# )

# plt.legend()
# plt.show()

model_5 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model_5.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

# print('without shaping: ',y)
# print('X shape: ', X, y.reshape((-1,1)))

# print('tanpa expand: ', X)
# print('bentuk: ', X.shape)
# print('lepas expand: ', tf.expand_dims(X, axis=-1))
# print('bentuk lepas: ', tf.expand_dims(X, axis=-1).shape)

# history = model_5.fit(tf.expand_dims(X, axis=-1), y.reshape((-1,1)), epochs=100)
history = model_5.fit(X, y, epochs=100)

# print('after shaping: ', y.reshape((-1,1)))

print(model_5.summary())

plot_decision_boundary(model_5, X, y)


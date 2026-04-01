# Solving XOR problem using deep feed forward network

import numpy as np
import tensorflow as tf
model = tf.keras.Sequential([
# tf.keras.layers.Dense(2, activation='relu', input_dim=2),
tf.keras.layers.Dense(2, activation='relu', input_shape=(2,)),
tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
x=np.array([[0.,0.],
[0.,1.],
[1.,0.],
[1.,1]], dtype=float)
y = np.array([0.,1.,1.,0.], dtype=float)
model.fit(x,y, epochs=1000, batch_size=4, verbose=0)

print("\nWeights After training:")
print(model.get_weights())
print("\nPredictions:")
print(model.predict(x))

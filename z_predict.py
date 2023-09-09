import tensorflow as tf
import numpy as np

loaded_model = tf.keras.models.load_model('model')

res = loaded_model.predict(np.array([[1.3,4.42]]))

print(res)
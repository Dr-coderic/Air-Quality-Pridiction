import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv("air-quality-india.csv")

print(df.head())

pm2_5 = df['PM2.5']
split_time = 6900
x_train = pm2_5[:split_time]
time_train = [i for i in range(0, split_time)]
time_valid = [i for i in range(split_time, len(pm2_5))]
x_valid = pm2_5[split_time:]

window_size = 24
batch_size = 16
shuffle_buffer_size = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


dataset = windowed_dataset(
    x_train, window_size, batch_size, shuffle_buffer_size)


tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                           input_shape=[None]),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Dense(1)
])


model.compile(loss=tf.keras.losses.Huber(),
              optimizer='adam',
              metrics=["mae"])
history = model.fit(dataset, epochs=1)

model.save('model')

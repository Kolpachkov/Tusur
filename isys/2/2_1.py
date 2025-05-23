import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers

# Загрузка полной модели
autoencoder = load_model("helicopter_model.h5", compile=False)

# Извлечение декодера из autoencoder
# Предпоследний слой был Dense(64), значит input -> этот вектор
latent_input = tf.keras.Input(shape=(64,))
x = autoencoder.layers[-7](latent_input)      # Dense(32*32*16)
x = autoencoder.layers[-6](x)                 # Reshape
x = autoencoder.layers[-5](x)                 # Conv2D
x = autoencoder.layers[-4](x)                 # UpSampling2D
x = autoencoder.layers[-3](x)                 # Conv2D
x = autoencoder.layers[-2](x)                 # UpSampling2D
x = autoencoder.layers[-1](x)                 # Conv2D (output)

decoder = Model(latent_input, x)

# Генерация случайного вектора и получение изображения
z = np.random.normal(size=(1, 64)).astype(np.float32)
generated = decoder.predict(z)

# Отображение
plt.imshow(generated[0])
plt.axis('off')
plt.title("Generated image")
plt.show()

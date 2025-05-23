import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Путь к сохранённой модели
decoder_path = 'vae_decoder.h5'

# Размерность латентного пространства
latent_dim = 64

# Загрузка декодера
decoder = load_model(decoder_path, compile=False)

# Генерация случайного латентного вектора
random_vector = np.random.normal(size=(1, latent_dim)).astype(np.float32)

# Генерация изображения
generated_image = decoder.predict(random_vector)

# Отображение
plt.imshow(generated_image[0])
plt.axis('off')
plt.title("Generated Image")
plt.show()

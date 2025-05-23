import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Загрузка датасета
dataset, info = tfds.load('plant_leaves', split='train', with_info=True, as_supervised=True)

# Предобработка
def preprocess(img, label):
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.float32) / 255.0
    return img, img  # input = output

dataset = dataset.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Сверточный автоэнкодер
input_img = layers.Input(shape=(128, 128, 3))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Обучение
autoencoder.fit(dataset, epochs=20)

# Сохранение модели
autoencoder.save("plant_leaves_autoencoder.h5")
print("Модель сохранена в 'plant_leaves_autoencoder.h5'")

# Визуализация
for batch in dataset.take(1):
    original, _ = batch
    reconstructed = autoencoder.predict(original)

    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(original[i])
        plt.axis('off')
        plt.subplot(2, 5, i + 6)
        plt.imshow(reconstructed[i])
        plt.axis('off')
    plt.show()

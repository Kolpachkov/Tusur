import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Параметры
latent_dim = 100
img_size = 64
channels = 3
img_shape = (img_size, img_size, channels)
batch_size = 64
epochs = 10000
save_interval = 1000
data_dir = "C:/Users/multi/Desktop/Helicopter Class 1"

# Загрузка изображений из папки
def load_images_from_folder(folder, target_size):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = load_img(path, target_size=target_size)
            img = img_to_array(img)
            images.append(img)
        except:
            continue
    images = np.array(images, dtype=np.float32)
    images = (images - 127.5) / 127.5  # Нормализация [-1, 1]
    return images

X_train = load_images_from_folder(data_dir, (img_size, img_size))

# Генератор
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, input_dim=latent_dim),
        layers.Reshape((8, 8, 256)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2DTranspose(channels, 4, strides=2, padding='same', activation='tanh')
    ])
    return model

# Дискриминатор
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 4, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Инициализация
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss='binary_crossentropy', metrics=['accuracy'])

# GAN-модель (обучение генератора)
discriminator.trainable = False
z = layers.Input(shape=(latent_dim,))
img = generator(z)
validity = discriminator(img)
combined = tf.keras.Model(z, validity)
combined.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                 loss='binary_crossentropy')

# Генерация изображений
def save_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # [0, 1]
    fig, axs = plt.subplots(r, c, figsize=(5, 5))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.tight_layout()
    plt.savefig(f"gan_output_{epoch}.png")
    plt.close()

# Обучение
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(1, epochs + 1):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined.train_on_batch(noise, valid)

    if epoch % 100 == 0:
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

    if epoch % save_interval == 0:
        save_images(epoch)

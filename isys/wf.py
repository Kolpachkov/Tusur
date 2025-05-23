import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image

# Параметры (задаются в одном месте)
image_size = 32      # Размер изображения (ширина=высота)
color_depth = 3       # Глубина цвета (3 = RGB, 1 = grayscale)
noise_dim = 100
batch_size = 256
epochs = 5000
BUFFER_SIZE = 10000

# Загрузка и подготовка данных
ds = load_dataset("James-A/Minecraft-16x-Dataset", split="train")

def preprocess(example):
    img = example["image"].convert("RGB" if color_depth == 3 else "L") \
        .resize((image_size, image_size), Image.NEAREST)
    img = np.array(img).astype("float32")
    # если grayscale, добавим размерность канала
    if color_depth == 1:
        img = np.expand_dims(img, axis=-1)
    img = (img - 127.5) / 127.5  # Нормализация [-1, 1]
    return img

images = []
for ex in ds:
    try:
        img = preprocess(ex)
        if img.shape == (image_size, image_size, color_depth):
            images.append(img)
    except:
        continue

images = np.array(images)
train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(batch_size)

# Генератор
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense((image_size // 4) * (image_size // 4) * 128, use_bias=False, input_shape=(noise_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((image_size // 4, image_size // 4, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(color_depth, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Дискриминатор
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=[image_size, image_size, color_depth]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Потери и оптимизаторы
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(real_output), real_output) + \
           cross_entropy(tf.zeros_like(fake_output), fake_output)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = (predictions[i] + 1) / 2.0
        if color_depth == 1:
            plt.imshow(img[:, :, 0], cmap='gray')
        else:
            plt.imshow(img)
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close()

def train(dataset, epochs):
    seed = tf.random.normal([16, noise_dim])
    for epoch in range(epochs):
        gen_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()
        for image_batch in dataset:
            # Выполняем шаг обучения
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                noise = tf.random.normal([image_batch.shape[0], noise_dim])
                generated_images = generator(noise, training=True)
                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            gen_loss_avg.update_state(gen_loss)
            disc_loss_avg.update_state(disc_loss)

        print(f"Epoch {epoch+1}/{epochs}, Generator loss: {gen_loss_avg.result():.4f}, Discriminator loss: {disc_loss_avg.result():.4f}")

    # Сохраняем изображение только после последней эпохи
    generate_and_save_images(generator, epochs, seed)

# Запуск обучения
train(train_dataset, epochs)

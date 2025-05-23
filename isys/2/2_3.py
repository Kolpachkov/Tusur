import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from datasets import load_dataset

# Загрузка датасета
dataset_info = load_dataset("James-A/Minecraft-16x-Dataset", split="train")

# Генератор для tf.data.Dataset, с конвертацией в RGB (3 канала)
def generator():
    for ex in dataset_info:
        img = ex['image'].convert("RGB")  # Убираем альфа-канал
        img = tf.convert_to_tensor(img)
        yield img, img

# Создаем tf.data.Dataset из генератора
tf_dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8))
)

# Предобработка: изменение размера и нормализация
def preprocess_image(image, label):
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    return image, image

tf_dataset = tf_dataset.map(preprocess_image)
tf_dataset = tf_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Параметры
input_shape = (128, 128, 3)
latent_dim = 256

# Sampling слой (репараметризация)
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

# Энкодер
inputs = layers.Input(shape=input_shape)
x = layers.Conv2D(64, 3, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling2D(2, padding='same')(x)

x = layers.Conv2D(128, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling2D(2, padding='same')(x)

x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.3)(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

# Декодер
decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(32 * 32 * 128)(decoder_inputs)
x = layers.LeakyReLU()(x)
x = layers.Reshape((32, 32, 128))(x)

x = layers.Conv2D(128, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.UpSampling2D(2)(x)

x = layers.Conv2D(64, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.UpSampling2D(2)(x)

x = layers.Conv2D(32, 3, padding='same')(x)
x = layers.LeakyReLU()(x)

outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
decoder = models.Model(decoder_inputs, outputs, name='decoder')

# VAE модель
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - reconstruction), axis=[1, 2, 3]))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(tf_dataset, epochs=10)

# Сохраняем модели
vae.encoder.save("_encoder.h5")
vae.decoder.save("_decoder.h5")

# Визуализация реконструкций
for batch in tf_dataset.take(1):
    original, _ = batch
    _, _, z = encoder(original)
    reconstructed = decoder(z)

    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(original[i])
        plt.axis('off')
        plt.subplot(2, 5, i + 6)
        plt.imshow(reconstructed[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
# Генерация случайного латентного вектора
random_vector = tf.random.normal(shape=(1, latent_dim)).numpy()
# Генерация изображения
generated_image = decoder.predict(random_vector)
# Отображение
plt.imshow(generated_image[0])
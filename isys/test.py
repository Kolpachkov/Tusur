import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# 1. Настройка GPU (если он доступен)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Ограничим использование памяти GPU по требованию
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU успешно настроен")
    except RuntimeError as e:
        print(e)
else:
    print("GPU не найден, используется CPU")

# 2. Загрузка датасета и информации о нем
dataset, info = tfds.load('plant_leaves', with_info=True)
val_dataset = dataset['train']  # Используем тренировочную выборку для демонстрации

# Список имен классов
class_names = info.features['label'].names

# 3. Функция предобработки изображений
def preprocess_image(example):
    image = example['image']
    label = example['label']
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

batch_size = 32
val_ds = val_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 4. Загрузка заранее обученной модели
model_path = "C:/Users/multi/Tusur/saved_models/plant_leaves_model.h5"
saved_model = tf.keras.models.load_model(model_path)

# 5. Демонстрация распознавания с выводом названий классов
# Берём один батч из валидационной выборки
test_images, test_labels = next(iter(val_ds))
predictions = saved_model.predict(test_images)

# Визуализируем несколько результатов
num_images = min(9, test_images.shape[0])
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()
for i in range(num_images):
    axes[i].imshow(test_images[i])
    pred_class = np.argmax(predictions[i])
    # Получаем имена классов для предсказанного и реального значений
    pred_name = class_names[pred_class]
    true_name = class_names[test_labels[i].numpy()]
    axes[i].set_title(f"Реальный: {true_name}\nПредсказанный: {pred_name}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()

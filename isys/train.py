import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os

# 1. Настройка GPU (если он доступен)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Ограничим использование памяти GPU по требованию
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Если нужно использовать только один GPU, раскомментируйте:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        print("GPU успешно настроен")
    except RuntimeError as e:
        print(e)
else:
    print("GPU не найден, используется CPU")

# 2. Загрузка датасета и информации о нем
dataset, info = tfds.load('plant_leaves', with_info=True)
train_dataset = dataset['train']

# 3. Разделение данных на обучающую и валидационную выборки (80/20)
train_split = 0.8
train_count = int(info.splits['train'].num_examples * train_split)
train_ds = train_dataset.take(train_count)
val_ds = train_dataset.skip(train_count)

# 4. Функция предобработки изображений
def preprocess_image(example):
    image = example['image']
    label = example['label']
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

batch_size = 32
train_ds = train_ds.map(preprocess_image).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 5. Создание модели на базе MobileNetV2
mobilenet = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
mobilenet.trainable = False

model = Sequential([
    mobilenet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(info.features['label'].num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Настройка callback для сохранения модели (сохранится лучшая по валидационной точности)
checkpoint_dir = "saved_models"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "plant_leaves_model.h5")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy')

# 7. Обучение модели
history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[checkpoint_cb])

# 8. Оценка модели на валидационной выборке
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Точность на валидационной выборке: {val_accuracy:.4f}")

# 9. Сохранение модели окончательно (если не использовали callback, можно вызвать model.save)
model.save(checkpoint_path)
print(f"Модель сохранена по пути: {checkpoint_path}")

# 10. Демонстрация распознавания
# Загружаем сохранённую модель
saved_model = tf.keras.models.load_model(checkpoint_path)

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
    axes[i].set_title(f"Реальный: {test_labels[i].numpy()} | Предсказанный: {pred_class}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()
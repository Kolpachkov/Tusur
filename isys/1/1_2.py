#1 координаты точки пересечения точки 2 прямых 2 мерное пространство в ограниченной области с помощью обратного распространения ошибки
# Mobile net
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions
import matplotlib.pyplot as plt

def read_image_files(path, target_size=(224, 224)):
    file_list = [file.path for file in os.scandir(path) if file.is_file() and file.name.endswith('.jpg')]
    
    if not file_list:
        raise ValueError("В указанной директории нет файлов .jpg")
    
    images = []
    for file in file_list:
        img = Image.open(file).convert("RGB")  
        img = img.resize(target_size, Image.LANCZOS)
        images.append(np.array(img, dtype=np.float32) / 255.0)
    
    return np.array(images)

image_path = "C:/Users/multi/Desktop/Tusur/isys/images"
image_box = read_image_files(image_path)
files_count = len(image_box)

mobilenet = MobileNetV2(weights="imagenet", include_top=True)

predictions = mobilenet.predict(image_box)

decoded_predictions = decode_predictions(predictions, top=1)
print(decoded_predictions)
for i, prediction in enumerate(decoded_predictions):
    print(f"Изображение {i + 1}: {prediction}")

fig, axes = plt.subplots(int(np.ceil(np.sqrt(files_count))), int(np.ceil(np.sqrt(files_count))), figsize=(10, 10))
axes = axes.flatten()

for i, img in enumerate(image_box):
    axes[i].imshow(img)
    axes[i].set_title(f"Prediction: {decoded_predictions[i][0][1]} = {decoded_predictions[i][0][2]:.2f}")
    axes[i].axis("off")
plt.show()

for layer in mobilenet.layers:
    print(layer.name, layer.output.shape)

model_mobilenet = tf.keras.models.Model(inputs=mobilenet.input, outputs=[
    mobilenet.get_layer("global_average_pooling2d").output,
    mobilenet.get_layer("block_5_expand").output,
    mobilenet.get_layer("block_10_expand").output,
    mobilenet.get_layer("block_15_expand").output
])

output = model_mobilenet.predict(image_box[:2])

fig1, ax1 = plt.subplots()
x = np.arange(len(output[0][0]))
ax1.plot(x, output[0][0], label="Image 1")
ax1.plot(x, output[0][1], label="Image 2")
ax1.legend()
plt.show()

out_img = output[1][0].transpose((2, 0, 1))
fig2, axes2 = plt.subplots(int(np.ceil(np.sqrt(out_img.shape[0]))), int(np.ceil(np.sqrt(out_img.shape[0]))), figsize=(12, 12))
axes2 = axes2.flatten()

for i in range(out_img.shape[0]):
    axes2[i].imshow(out_img[i], cmap="gray")
    axes2[i].axis("off")

plt.show()


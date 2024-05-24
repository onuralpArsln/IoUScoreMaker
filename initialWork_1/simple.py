import cv2
import numpy as np
import tensorflow as tf



model_path = 'home/onuralp/projects/Segmentasyon/ModelRunner/best_model.hdf5'
model = tf.keras.models.load_model(model_path)

# Giriş resmini uygun boyuta getirme
input_image_path = 'home/onuralp/projects/Segmentasyon/ModelRunner/002.jpg'
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (256, 256))  # Giriş boyutunu (256, 256) olarak yeniden boyutlandırma
input_image = np.expand_dims(input_image, axis=-1)  # Tek bir kanal ekleyerek (256, 256) boyutunu (256, 256, 1) olarak genişletiyoruz
# Tahmin yapma
prediction_result = model.predict(np.expand_dims(input_image, axis=0))

# Numpy dizisini JPEG dosyasına dönüştürme
output_image_path = 'prediction_result1.jpg'
prediction_result = (prediction_result * 255).astype(np.uint8)
cv2.imwrite(output_image_path, prediction_result[0])

# Dönüştürülen JPEG dosyasını siyah beyaz hale getirme
output_image_gray_path = 'prediction_result.jpg'
output_image_gray = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
cv2.imwrite(output_image_gray_path, output_image_gray)

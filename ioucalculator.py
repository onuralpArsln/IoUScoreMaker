import cv2
import numpy as np
import tensorflow as tf

# Özel metrik fonksiyonları tanımlama
def dice_coef(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + 1e-5) / (union + 1e-5)

def iou_score(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-5) / (union + 1e-5)

# Modeli yükleme sırasında özel metrik fonksiyonları tanımlama
with tf.keras.utils.custom_object_scope({'dice_coef': dice_coef, 'iou_score': iou_score}):
    model_path = 'D:\\linmig\\best_model.hdf5'
    model = tf.keras.models.load_model(model_path)




# Giriş resmini uygun boyuta getirme
input_image_path = 'D:/linmig/002.jpg'
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (256, 256))  # Giriş boyutunu (256, 256) olarak yeniden boyutlandırma
input_image = np.expand_dims(input_image, axis=-1)  # Tek bir kanal ekleyerek (256, 256) boyutunu (256, 256, 1) olarak genişletiyoruz
# Tahmin yapma
prediction_result = model.predict(np.expand_dims(input_image, axis=0))
side_res=prediction_result
# Numpy dizisini JPEG dosyasına dönüştürme
output_image_path = 'prediction_result1.jpg'
prediction_result = (prediction_result * 255).astype(np.uint8)
cv2.imwrite(output_image_path, prediction_result[0])

# Dönüştürülen JPEG dosyasını siyah beyaz hale getirme
output_image_gray_path = 'prediction_result.jpg'
output_image_gray = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
cv2.imwrite(output_image_gray_path, output_image_gray)



#maske pathi ayarla 
input_mask_path = 'D:\linmig\mask_002.png'
input_mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
input_mask = cv2.resize(input_mask, (256, 256))
input_mask = np.expand_dims(input_mask, axis=-1) 
input_mask = input_mask.astype(np.float32) / 255.0


def read_matlab_annotations(file_path):
    # Your code to read MATLAB annotations and convert to binary masks
    # Example:
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
    return mask

mask2 = read_matlab_annotations("D:\linmig\mask_002.png")
mask2 = cv2.resize(mask2, (256, 256))
mask2 = np.expand_dims(mask2, axis=-1) 
mask2 = mask2.astype(np.float32) / 255.0

def threshold_mask(mask, threshold=0.5):
    return (mask > threshold).astype(np.uint8)

predicted_mask = threshold_mask(side_res)

# Normalize masks
ground_truth_mask = mask2.astype(np.float32) / 255.0
predicted_mask = predicted_mask.astype(np.float32) / 255.0



reverse_res = prediction_result.astype(np.float32) / 255.0
iou = iou_score(reverse_res, input_mask)

iou = iou_score(ground_truth_mask, predicted_mask)

print("IoU Score:", iou)
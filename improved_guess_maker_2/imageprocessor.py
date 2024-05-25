# this image process make brightnes 3.4 and contrast 0.4 to improve readibility of image 


import cv2
import numpy as np
import tensorflow as tf

def imgProces( img : np.ndarray) -> np.ndarray:
    alpha = 0.4  # Contrast control (1.0 means no change)
    beta = 2.4    # Brightness control (0 means no change)
    adjusted_image = cv2.convertScaleAbs(input_image, alpha=alpha, beta=beta)
    return adjusted_image
  
# test part 
if __name__ == "__main__":
    input_image_path = 'C:\\Users\\onura\\Documents\\VSproject\\linmig\\improved_guess_maker_2\\002.jpg'
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (256, 256))  # Giriş boyutunu (256, 256) olarak yeniden boyutlandırma
    input_image = np.expand_dims(input_image, axis=-1)  # Tek bir kanal ekleyerek (256, 256) boyutunu (256, 256, 1) olarak genişletiyoruz

    result = imgProces(input_image)

    result_path = 'C:\\Users\\onura\\Documents\\VSproject\\linmig\\improved_guess_maker_2\\002res.jpg'
    result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(result_path, result)


    proces_path = 'C:\\Users\\onura\\Documents\\VSproject\linmig\\002.jpg'
    proces_image = cv2.imread(proces_path, cv2.IMREAD_GRAYSCALE)
    proces_image = cv2.resize( proces_image, (256, 256))  # Giriş boyutunu (256, 256) olarak yeniden boyutlandırma
    proces_image = np.expand_dims( proces_image, axis=-1)  # Tek bir kanal ekleyerek (256, 256) boyutunu (256, 256,




    threshold_value = 20

    # Binarize the image using the threshold
  
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
        model_path = 'C:\\Users\\onura\\Documents\\VSproject\\linmig\\best_model.hdf5'
        model = tf.keras.models.load_model(model_path)

    prediction_result = model.predict(np.expand_dims(input_image, axis=0))
    prediction_result = (prediction_result * 255).astype(np.uint8)

    prediction_result2 = model.predict(np.expand_dims(proces_image, axis=0))
    prediction_result2 = (prediction_result * 255).astype(np.uint8)

    # Display the original and binary images
    cv2.imshow('Original Image', input_image)
    cv2.imshow('Processed Image', result)
    cv2.imshow('Guessed Image', prediction_result[0])
    cv2.imshow('Guessed  Processed Image', prediction_result2[0])
    cv2.imwrite('./aiImage.png', prediction_result2[0]) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
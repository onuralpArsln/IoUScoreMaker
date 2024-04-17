import cv2
import numpy as np
import tensorflow as tf
import graphmaker2d as gmaker
import ioucalculator as calcul

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
input_image_path = 'D:\\linmig\\002.jpg'
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (256, 256))  # Giriş boyutunu (256, 256) olarak yeniden boyutlandırma
input_image = np.expand_dims(input_image, axis=-1)  # Tek bir kanal ekleyerek (256, 256) boyutunu (256, 256, 1) olarak genişletiyoruz

gaus_kernel_x=3
gaus_kernel_y=3
gaus_kernel_std_dev=0
kernelSize=[]
iouScore=[]



for i in range(50):
    ### Burada Görüntü İşlemeleri Yap ↓↓↓↓↓↓
    # gaus_kerneller her zaman tek sayı olmalı çift sayıda hata verir 
    blurred_image = cv2.GaussianBlur(input_image, (gaus_kernel_x, gaus_kernel_y), gaus_kernel_std_dev+i)
    
    ### Burada Görüntü İşlemeleri Yap ^^^^^^^^^^^^^^
    # Tahmin yapma
    prediction_result = model.predict(np.expand_dims(blurred_image, axis=0))
    # Numpy dizisini JPEG dosyasına dönüştürme
    output_image_path = 'prediction_resulttemp.jpg'
    prediction_result = (prediction_result * 255).astype(np.uint8)
    cv2.imwrite(output_image_path, prediction_result[0])
    # Dönüştürülen JPEG dosyasını siyah beyaz hale getirme
    output_image_gray_path = 'prediction_result.jpg'
    output_image_gray = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
    #  input_image iou score için hazır boyutlu ve grayscale
    #  output_image_gray iou score için hazır boyutlu ve grayscale

     # Grafik için data toplama  
    kernelSize.append(gaus_kernel_std_dev+i)
    iou=calcul.iou(output_image_gray,input_image)
    iouScore.append(iou)


### Burada grafik isimlendirmesi oluyor  graphmaker2d.py dosyasında açıklaması var  ↓↓↓↓↓↓
gmaker.graph(kernelSize, iouScore ,"GLP - IOU Score normalized outputs Sigma Change while kernel is 11by11" ,xname="Gaussian Standart Dev")


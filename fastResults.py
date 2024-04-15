import cv2
import numpy as np
import tensorflow as tf
import graphmaker2d as gmaker


### dosya yolları 
maskPath = 'D:\linmig\mask_002.png'
modPath = 'D:\\linmig\\best_model.hdf5'
inputPath = 'D:/linmig/002.jpg'




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
    model_path = modPath
    model = tf.keras.models.load_model(model_path)


#maske pathi ayarla 
input_mask_path = maskPath 
input_mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
input_mask = cv2.resize(input_mask, (256, 256))
input_mask = np.expand_dims(input_mask, axis=-1) 
input_mask = input_mask.astype(np.float32) / 255.0


# Giriş resmini uygun boyuta getirme
input_image_path = inputPath
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (256, 256))  # Giriş boyutunu (256, 256) olarak yeniden boyutlandırma
input_image = np.expand_dims(input_image, axis=-1)  # Tek bir kanal ekleyerek (256, 256) boyutunu (256, 256, 1) olarak genişletiyoruz




gaus_kernel_x=5
gaus_kernel_y=5
gaus_kernel_std_dev=-20
kernelSize=[]
iouScore=[]



for i in range(50):
    ### Burada Görüntü İşlemeleri Yap ↓↓↓↓↓↓
    # gaus_kerneller her zaman tek sayı olmalı çift sayıda hata verir 
    blurred_image = cv2.GaussianBlur(input_image, (gaus_kernel_x, gaus_kernel_y), gaus_kernel_std_dev+i)
    sharp_image = cv2.addWeighted(input_image, 1.5, blurred_image, -0.5, 0)
    ### Burada Görüntü İşlemeleri Yap ↓^^^^^^^^^^^^^^
    
    # Grafik için data toplama  
    kernelSize.append(gaus_kernel_std_dev+i)

    #Model ile tahmin yapılması
    prediction_result = model.predict(np.expand_dims(sharp_image, axis=0))
    side_res=prediction_result
    
    # iou score hesabbı
    iou = iou_score(side_res, input_mask)

    # iou score agrafik için listeye eklendi .numpy() float almamızı sağlar
    iouScore.append(iou.numpy())


 ### Burada grafik isimlendirmesi oluyor  graphmaker2d.py dosyasında açıklaması var  ↓↓↓↓↓↓
gmaker.graph(kernelSize, iouScore ,"GLP and Laplacian Sharpened - IOU Score" ,xname="Gaussian  Standart Deviation")



    

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


#maske pathi ayarla 
input_mask_path = 'D:\linmig\mask_002.png'
input_mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
input_mask = cv2.resize(input_mask, (256, 256))
input_mask = np.expand_dims(input_mask, axis=-1) 
input_mask = input_mask.astype(np.float32) / 255.0


# Giriş resmini uygun boyuta getirme
input_image_path = 'D:/linmig/002.jpg'
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (256, 256))  # Giriş boyutunu (256, 256) olarak yeniden boyutlandırma
input_image = np.expand_dims(input_image, axis=-1)  # Tek bir kanal ekleyerek (256, 256) boyutunu (256, 256, 1) olarak genişletiyoruz




gaus_kernel_x=5
gaus_kernel_y=5
gaus_kernel_std_dev=-20
blurred_image = cv2.GaussianBlur(input_image, (gaus_kernel_x, gaus_kernel_y), gaus_kernel_std_dev)

kernelSize=[]
iouScore=[]

#cv2.imshow('Original Image', input_image)

for i in range(50):
    # use i*2 to keep kernel side length always odd number
    blurred_image = cv2.GaussianBlur(input_image, (gaus_kernel_x, gaus_kernel_y), gaus_kernel_std_dev+i)
    sharp_image = cv2.addWeighted(input_image, 1.5, blurred_image, -0.5, 0)
    #cv2.imshow(f'Blurred Image {i}', blurred_image)
    
    # kernel size data collected 
    kernelSize.append(gaus_kernel_std_dev+i)

    #prediction made
    prediction_result = model.predict(np.expand_dims(sharp_image, axis=0))
    #prediction tuned into value 
    side_res=prediction_result
    
    # calculate iou score
    iou = iou_score(side_res, input_mask)
    # iou score added to list
   
    iouScore.append(iou.numpy())

import graphmaker2d as gmaker

gmaker.graph(kernelSize, iouScore ,"GLP and Laplacian Sharpened - IOU Score" ,xname="Gaussian  Standart Deviation")



    

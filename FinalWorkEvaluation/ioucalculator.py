import cv2
import numpy as np

def iou(mask1, mask2):
    _, mask1 = cv2.threshold(mask1, 128, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(mask2, 128, 255, cv2.THRESH_BINARY)
    mask1 = mask1 / 255
    mask2 = mask2 / 255
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


if __name__ == "__main__":

    def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    # Load your grayscale images
    image1 = cv2.imread("D:\\linmig\\002after.jpg", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("D:\\linmig\\002after.jpg", cv2.IMREAD_GRAYSCALE)

    # Threshold the images to create binary masks
    _, mask1 = cv2.threshold(image1, 128, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(image2, 128, 255, cv2.THRESH_BINARY)

    # Normalize mask values to 0 and 1
    mask1 = mask1 / 255
    mask2 = mask2 / 255

    # Calculate IOU score
    iou_score = calculate_iou(mask1, mask2)

    print("IOU score:", iou_score)




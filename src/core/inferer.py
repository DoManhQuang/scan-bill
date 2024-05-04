import tempfile
import numpy as np
import cv2
from utils.images_processing import image_warped_transform
from utils.engine import ultra_keypoints


def scan_bill(source, infer_weights):
    ppocr_infer, bill_infer, money_infer = infer_weights
    image_bill = cv2.imread(source)
    print("image_bill: ", image_bill.shape)
    results_bill = bill_infer.predict(image_bill, save=False, conf=0.7, iou=0.7, imgsz=640)
    _, _, _, bill_keypts = ultra_keypoints(results_bill)
    bill_imagew = image_warped_transform(image=image_bill, keypoint=bill_keypts, rescale_intensity=False)
    results_money = money_infer.predict(bill_imagew, save=True, conf=0.7, iou=0.7, imgsz=640)
    _, _, _, money_keypts = ultra_keypoints(results_money)
    money_imagew = image_warped_transform(image=bill_imagew, keypoint=money_keypts, rescale_intensity=False)
    text = ppocr_infer.ocr(money_imagew, rec=True, cls=True, det=False)[0]
    return text[0][0], text[0][1]



    
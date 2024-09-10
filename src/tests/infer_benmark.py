from paddleocr.ppstructure.utility import parse_args
from paddleocr.ppstructure.kie.predict_kie_token_ser import SerPredictor
import numpy as np
import cv2, json, sys
from utils.images_processing import image_warped_transform
from utils.engine import ultra_keypoints
import os 
from paddleocr.ppocr.utils.visual import draw_ser_results
from paddleocr.ppocr.utils.utility import get_image_file_list, check_and_read
from paddleocr.ppocr.utils.logging import get_logger
import time
from utils.engine import ultra_model_load


ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

logger = get_logger()


def kie_ser_prediction(args, ser_systems, keypoint_infer, save=True, draw=False):
    image_file_list = get_image_file_list(args.image_dir)
    count = 0
    total_time = []
    os.makedirs(args.output, exist_ok=True)
    with open(
        os.path.join(args.output, "infer.txt"), mode="w", encoding="utf-8"
    ) as f_w:
        for image_file in image_file_list:
            try:
                img, flag, _ = check_and_read(image_file)
                if not flag:
                    img = cv2.imread(image_file)
                    img = img[:, :, ::-1]
                if img is None:
                    logger.info("error in loading image:{}".format(image_file))
                    continue
                
                start_time = time.time()
                # yolo detect keypoint
                results_bill = keypoint_infer.predict(img, save=False, conf=0.7, iou=0.7, imgsz=640)
                _, _, _, bill_keypts = ultra_keypoints(results_bill)
                new_imagew = image_warped_transform(image=img, keypoint=bill_keypts, rescale_intensity=False)

                # layoutxlm detect layout
                ser_res, _, elapse = ser_systems(new_imagew)
                ser_res = ser_res[0]
                end_time = time.time()
                if save:
                    res_str = "{}\t{}\n".format(
                        image_file,
                        json.dumps(
                            {
                                "ocr_info": ser_res,
                            },
                            ensure_ascii=False,
                        ),
                    )
                    f_w.write(res_str)

                    if draw:
                        img_res = draw_ser_results(
                            image_file,
                            ser_res,
                            font_path=args.vis_font_path,
                        )

                        img_save_path = os.path.join(args.output, os.path.basename(image_file))
                        cv2.imwrite(img_save_path, img_res)
                        logger.info("save vis result to {}".format(img_save_path))
                if count > 5:
                    total_time.append((end_time - start_time))
                count += 1
                logger.info("Predict KIE LaypoutXLM time of {}: {}".format(image_file, elapse))
                logger.info("Predict pipline time of {}: {}".format(image_file, (end_time - start_time)))
            except Exception as e:
                logger.debug("ERROR >>> {}".format(e))
                logger.debug("ERROR >>> image path: {}".format(image_file))
    return ser_res, total_time


serargs = parse_args()
serargs.kie_algorithm="LayoutXLM"
serargs.vis_font_path="/work/quang.domanh/ppocr/ppocr-vietnamese/paddleocr/inference/gpkd_weights/RobotoSlab-Light.ttf"
serargs.use_gpu=False
serargs.ser_model_dir="/work/quang.domanh/scan-bill/src/weights/kie/invoices/ser_layoutxlm_invoices_fold1"
serargs.ser_dict_path="/work/quang.domanh/scan-bill/src/assets/invoices_classes.txt"
serargs.ocr_order_method="tb-yx"
serargs.use_angle_cls="/work/quang.domanh/scan-bill/src/weights/ocr/cls/ch_ppocr_mobile_v2.0_cls_infer"
serargs.det_model_dir="/work/quang.domanh/scan-bill/src/weights/ocr/det/ch/ch_PP-OCRv4_det_infer"
serargs.rec_model_dir="/work/quang.domanh/scan-bill/src/weights/ocr/rec/ch/ch_PP-OCRv4_rec_infer"
serargs.image_dir="/work/quang.domanh/datasets/test_benmark_invoices"

ser_predictor = SerPredictor(serargs)
print("load model  ser layoutxlm done!")
bill_keypts_infer = ultra_model_load(weights_path="/work/quang.domanh/scan-bill/src/weights/yolo/bill-yolov8s-pose-best.pt")
print("load model  bill_keypts_infer done!")


ser_res, total_time = kie_ser_prediction(args=serargs, ser_systems=ser_predictor, keypoint_infer=bill_keypts_infer, save=True, draw=False)

# print(total_time)
print("benmark total: ", len(total_time))
print("mean time: {} image/s".format(np.mean(total_time)))
print("median time: {} image/s".format(np.median(total_time)))
print("P99 time: {} image/s".format(np.percentile(total_time, 99)))
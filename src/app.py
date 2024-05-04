import os, cv2
import sys
import time
from flask import Flask, request
from flask_restful import Resource, Api
from paddleocr import PaddleOCR
import gradio as gr


ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from core.inferer import scan_bill
from utils.engine import ultra_model_load

smart_banking = ['agribank', 'vcb', 'bidv', 'LP', 'viettel_money', 'ocb', 'tcb', 'vtb', 'mb', 'vib_ver2', 'acb', 'tp', 'vpbank', 'msb']
money_keypts_infer_list = {}
ppocr_infer = PaddleOCR(
    use_angle_cls=True, use_gpu=True, 
    det_model_dir="weights/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer",
    rec_model_dir="weights/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer",
    cls_model_dir="weights/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer"
)
print("load model  ppocr_infer done!")

bill_keypts_infer = ultra_model_load(weights_path="weights/bill/bill-yolov8s-pose-best.pt")
print("load model  bill_keypts_infer done!")

for bank in smart_banking:
    money_keypts_infer = ultra_model_load(weights_path=f"weights/smart-banking/{bank}/{bank}-yolov8s-pose-best.pt")
    money_keypts_infer_list[bank] = money_keypts_infer
    print(f"load model {bank} money_infer done!")


def demo_scan_bill(image, bank):
    temp_dir_input = "/data-gpu/quangdm2/OptiVisionLab/scan-bill/runs"
    save_path_input = os.path.join(temp_dir_input, "demo.jpg")
    cv2.imwrite(save_path_input, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print("save >> ", save_path_input)
    infer_weights = (ppocr_infer, bill_keypts_infer, money_keypts_infer_list[bank])
    text_money, text_conf = scan_bill(source=save_path_input, infer_weights=infer_weights)
    return f"AI: Độ tin cậy = {round((float(text_conf) * 100), 3)} %  và  Số Tiền = {text_money}"


demo = gr.Interface(fn=demo_scan_bill,
    inputs=[
        "image", 
        gr.Dropdown(
            smart_banking, label="Ngân Hàng", info="Chọn một trong các ngân hàng sau"
        )
    ], 
    outputs="textbox",
    examples=[
        ["/data-gpu/quangdm2/OptiVisionLab/scan-bill/src/assets/data.jpg", "vcb"],
        ["/data-gpu/quangdm2/OptiVisionLab/scan-bill/src/assets/data2.png", "bidv"],
    ]
    )
    

demo.launch(share=True, debug=True) 
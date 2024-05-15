from ultralytics import YOLO

model = YOLO('/data-gpu/quangdm2/OptiVisionLab/scan-bill/weights/bill/bill-yolov8s-pose-best.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.val(data='/data-gpu/quangdm2/OptiVisionLab/bill_keypoint_landmarks/data.yaml', 
                    batch=8, imgsz=680, device=0, iou=0.7, conf=0.7,
                    project="runs/bill_printed/bill_all/yolov8s_pose")
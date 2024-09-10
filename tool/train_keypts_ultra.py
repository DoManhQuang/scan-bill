from ultralytics import YOLO

model = YOLO('yolov8m-pose.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/data-gpu/quangdm2/OptiVisionLab/datasets/bill_keypoint_landmarks/data.yaml', epochs=200, imgsz=680, device=0, 
                      optimizer='AdamW', lr0=1e-4, batch=8, cos_lr=True, pretrained=True, project="runs/bill_printed/bill_all/yolov8m_pose", 
                      close_mosaic=10, degrees=90)
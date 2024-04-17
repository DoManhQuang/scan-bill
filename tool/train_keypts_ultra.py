from ultralytics import YOLO

model = YOLO('yolov8m-pose.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/home/fit/optivisionlab/scan-bill/bill_printed/vib_ver2/image_warped/data.yaml', epochs=100, imgsz=680, device='cpu', 
                      optimizer='SGD', lr0=1e-4, batch=8, cos_lr=True, pretrained=True, project="runs/bill_printed/money/yolov8m_pose", 
                      close_mosaic=10, degrees= 90, pose = 12, shear=0.3, scale=0.2)
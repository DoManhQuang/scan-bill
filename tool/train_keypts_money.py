from ultralytics import YOLO
import pathlib, os


def train(model_path='yolov8s-pose.pt', file_yml='data.yaml', project_name='runs'):
    model = YOLO(model_path)  # load a pretrained model (recommended for training)

    save_train = pathlib.Path(model_path).stem
    # Train the model
    results = model.train(data=file_yml, epochs=200, imgsz=680, device=0, 
                          optimizer='AdamW', lr0=1e-4, batch=8, cos_lr=True, pretrained=True, 
                          project=f"runs/bill_printed/{project_name}/{save_train}", 
                          close_mosaic=10, degrees= 90)
    return results


root = "/data-gpu/quangdm2/OptiVisionLab/money_keypoint_landmarks"
dirs = os.listdir(root)
for dir in dirs:
    print(f"START >>>>>>>>>> {dir} <<<<<<<<<<")

    results = train(model_path='yolov8s-pose.pt', file_yml=os.path.join(root, dir, 'data.yaml'), project_name=f"money/{dir}")
    print(f"results train {dir} : {results}")

    print(f"END >>>>>>>>>> {dir} <<<<<<<<<<")
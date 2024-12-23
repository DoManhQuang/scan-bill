from ultralytics import YOLO
import os, shutil, pathlib
import pandas as pd

def move_file(source_folder, destination_folder, file_name):
    # Check if the source file exists
    source_path = os.path.join(source_folder, file_name)
    if not os.path.exists(source_path):
        print(f"Source: {source_path}, file '{file_name}' not found in '{source_folder}'.")
        return
    
    # Check if the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)  # Create the destination folder if it doesn't exist
    
    # Construct the destination path
    destination_path = os.path.join(destination_folder, file_name)
    
    try:
        shutil.move(source_path, destination_path)
        print(f"File '{file_name}' moved from '{source_folder}' to '{destination_folder}'.")
    except Exception as e:
        print(f"Failed to move the file: {e}")

models_list = ['yolo11m-pose.pt']

source_dir = "/work/quang.domanh/datasets/bill_keypoint_landmarks/dataset"
target_dir = "/work/quang.domanh/datasets/bill_keypoint_landmarks"

for model_name in models_list:
    for i in range(2, 5):
        train_df = pd.read_csv(f"/work/quang.domanh/datasets/bill_keypoint_landmarks/kfold/train_fold_{i}.csv")
        val_df = pd.read_csv(f"/work/quang.domanh/datasets/bill_keypoint_landmarks/kfold/val_fold_{i}.csv")
        
        # setup data 
        for train_file in train_df.values.T[0]:
                move_file(source_folder=os.path.join(source_dir, "images"), 
                        destination_folder=os.path.join(target_dir, "images/train"), 
                        file_name=os.path.basename(train_file)
                        )
                move_file(source_folder=os.path.join(source_dir, "labels"), 
                        destination_folder=os.path.join(target_dir, "labels/train"), 
                        file_name=pathlib.Path(os.path.basename(train_file)).stem + '.txt'
                        )
        
        for val_file in val_df.values.T[0]:
                move_file(source_folder=os.path.join(source_dir, "images"), 
                        destination_folder=os.path.join(target_dir, "images/val"), 
                        file_name=os.path.basename(val_file)
                        )
                move_file(source_folder=os.path.join(source_dir, "labels"), 
                        destination_folder=os.path.join(target_dir, "labels/val"), 
                        file_name=pathlib.Path(os.path.basename(val_file)).stem + '.txt'
                        )
        
        model = YOLO(model_name)  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='/work/quang.domanh/datasets/bill_keypoint_landmarks/data.yaml', epochs=100, imgsz=680, device=0, 
                            optimizer='AdamW', lr0=1e-4, batch=64, cos_lr=True, pretrained=True, project=f"runs/keydet/{model_name}/fold_{i}", 
                            close_mosaic=10, degrees=15)
        
        # rollback data
        for train_file in train_df.values.T[0]:
                move_file(source_folder=os.path.join(target_dir, "images/train"), 
                        destination_folder=os.path.join(source_dir, "images"), 
                        file_name=os.path.basename(train_file)
                        )
                move_file(source_folder=os.path.join(target_dir, "labels/train"), 
                        destination_folder=os.path.join(source_dir, "labels"), 
                        file_name=pathlib.Path(os.path.basename(train_file)).stem + '.txt'
                        )
        
        for val_file in val_df.values.T[0]:
                move_file(source_folder=os.path.join(target_dir, "images/val"), 
                        destination_folder=os.path.join(source_dir, "images"),
                        file_name=os.path.basename(val_file)
                        )
                move_file(source_folder=os.path.join(target_dir, "labels/val"),
                        destination_folder=os.path.join(source_dir, "labels"), 
                        file_name=pathlib.Path(os.path.basename(val_file)).stem + '.txt'
                        )
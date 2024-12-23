from ultralytics import YOLO
import os, shutil, pathlib
import pandas as pd
import numpy as np


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


models_list = ["yolo11m-pose.pt", "yolo11n-pose.pt", "yolo11s-pose.pt", "yolov8m-pose.pt", "yolov8n-pose.pt", "yolov8s-pose.pt"]
save_dir_model = "/work/quang.domanh/scan-bill/runs/keydet"
source_dir = "/work/quang.domanh/datasets/bill_keypoint_landmarks/dataset"
target_dir = "/work/quang.domanh/datasets/bill_keypoint_landmarks"

with open("results-test-kfold.txt", "w") as f:
    f.write(">>> START <<< \n")

for model_name in models_list:

    print("TEST >>> ", model_name)
    scores95, score75, score50 = [], [], []
    keyscores95, keyscore75, keyscore50 = [], [], []

    with open("results-test-kfold.txt", "a") as f:
        f.write(">>> {} <<< \n". format(model_name))

    for i in range(0, 5):
        test_df = pd.read_csv(f"/work/quang.domanh/datasets/bill_keypoint_landmarks/kfold/test_fold_{i}.csv")
        
        # setup data 
        for test_file in test_df.values.T[0]:
                move_file(source_folder=os.path.join(source_dir, "images"), 
                        destination_folder=os.path.join(target_dir, "images/test"), 
                        file_name=os.path.basename(test_file)
                        )
                move_file(source_folder=os.path.join(source_dir, "labels"), 
                        destination_folder=os.path.join(target_dir, "labels/test"), 
                        file_name=pathlib.Path(os.path.basename(test_file)).stem + '.txt'
                        )
        
        model_path = os.path.join(save_dir_model, model_name, f"fold_{i}", "train", "weights", "best.pt")
        print(model_path)
        model = YOLO(model_path)  # load a pretrained model (recommended for training)

        metrics = model.val(data="/work/quang.domanh/datasets/bill_keypoint_landmarks/data.yaml", imgsz=640, batch=8, device="cpu",
                                       project=f"runs/keydet/{model_name}/test_fold_{i}", split="test", conf=0.5, iou=0.7)
        
        # print("metrics >>>> ", metrics)

        scores95.append(metrics.box.map)
        score75.append(metrics.box.map75)
        score50.append(metrics.box.map50)

        keyscores95.append(metrics.pose.map)
        keyscore75.append(metrics.pose.map75)
        keyscore50.append(metrics.pose.map50)

        with open("results-test-kfold.txt", "a") as f:
            f.write("TEST FOLD DET {} : mAP50-95 = {}, mAP75 = {}, mAP50 = {} \n".format(i, metrics.box.map, metrics.box.map75, metrics.box.map50))
            f.write("TEST FOLD KEY {} : mAP50-95 = {}, mAP75 = {}, mAP50 = {} \n".format(i, metrics.pose.map, metrics.pose.map75, metrics.pose.map50))
             
        # rollback data
        for test_file in test_df.values.T[0]:
                move_file(source_folder=os.path.join(target_dir, "images/test"), 
                        destination_folder=os.path.join(source_dir, "images"), 
                        file_name=os.path.basename(test_file)
                        )
                move_file(source_folder=os.path.join(target_dir, "labels/test"), 
                        destination_folder=os.path.join(source_dir, "labels"), 
                        file_name=pathlib.Path(os.path.basename(test_file)).stem + '.txt'
                        )
        
    with open("results-test-kfold.txt", "a") as f:
        f.write("\nDET AVG TEST: mAP50-95 = {}, mAP75 = {}, mAP50 = {} \n".format(np.average(scores95), np.average(score75), np.average(score50)))
        f.write("DET STD TEST: mAP50-95 = {}, mAP75 = {}, mAP50 = {} \n".format(np.std(scores95), np.std(score75), np.std(score50)))
        f.write("KEY AVG TEST: mAP50-95 = {}, mAP75 = {}, mAP50 = {} \n".format(np.average(keyscores95), np.average(keyscore75), np.average(keyscore50)))
        f.write("KEY STD TEST: mAP50-95 = {}, mAP75 = {}, mAP50 = {} \n\n".format(np.std(keyscores95), np.std(keyscore75), np.std(keyscore50)))
        f.write("END TEST >>> {} <<< \n\n". format(model_name))

    print("END TEST >>> ", model_name, "<<< \n")

with open("results-test-kfold.txt", "a") as f:
    f.write(">>> END <<< \n")
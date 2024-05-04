import pathlib, urllib, os, cv2, shutil
import glob, sys
import numpy as np
from ultralytics import YOLO


supported_file_types = ('.png', '.jpg')


def check_suffix_image(source, save_folder):
    img_name = os.path.basename(source)
    file_extension = pathlib.Path(img_name).suffix
    if file_extension in supported_file_types:
        return os.path.join(save_folder, img_name)
    return os.path.join(save_folder, f"{pathlib.Path(img_name).stem}.jpg")


def get_source_img(source, mode, save_folder):
    if mode == "path":
        # image = Image.open(source)
        image = cv2.imread(source)
        save_img = check_suffix_image(source=source, save_folder=save_folder)
        cv2.imwrite(save_img, image)
        # print("saved => ", save_img)
        return save_img
    elif mode == "url":
        save_img = check_suffix_image(source=source, save_folder=save_folder)
        urllib.request.urlretrieve(source, save_img)  # download images
        # print("saved => ", save_img)
        return save_img
    else:
        raise NotImplementedError(
            f"Mode {mode} is not supported. Should be path or url"
        )
    

def points2pointsn(points, original_height, original_width):
    pointn = []
    for x, y in points:
        xn = x / original_width
        yn = y / original_height
        pointn.append([xn, yn])
    return pointn

def normalize_keypoints(keypoints, width, height):
    normalized_keypoints = [(x / width, y / height) for x, y in keypoints]
    return normalized_keypoints


def xyxy_to_xywh_normalized(xyxy_bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = xyxy_bbox
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    x_center_normalized = x_center / image_width
    y_center_normalized = y_center / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height

    return x_center_normalized, y_center_normalized, width_normalized, height_normalized

def keypoints_to_bbox(keypoints):
    x_coordinates = [point[0] for point in keypoints]
    y_coordinates = [point[1] for point in keypoints]

    x_min = min(x_coordinates)
    y_min = min(y_coordinates)
    x_max = max(x_coordinates)
    y_max = max(y_coordinates)
    return x_min, y_min, x_max, y_max


def move_file(source_folder, destination_folder, file_name):
    # Check if the source file exists
    source_path = os.path.join(source_folder, file_name)
    if not os.path.exists(source_path):
        print(f"Source: {source_path}, file '{file_name}' not found in '{source_folder}'.")
        return
    
    # Check if the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # Create the destination folder if it doesn't exist
    
    # Construct the destination path
    destination_path = os.path.join(destination_folder, file_name)
    
    try:
        shutil.move(source_path, destination_path)
        print(f"File '{file_name}' moved from '{source_folder}' to '{destination_folder}'.")
    except Exception as e:
        print(f"Failed to move the file: {e}")


def bbox_to_keypoints(bbox):
    x_min, y_min, x_max, y_max = bbox
    keypoints = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    return keypoints


def ultra_model_load(weights_path):
    model = YOLO(weights_path)
    return model


def ultra_keypoints(inputs):
    '''
    input: results by ultralytics (only one image)
    output:
        conf: float
        boxes: [xmin, ymin, xmax, ymax]
        keypoints: [[x,y], [x,y], [x,y], [x,y]] max 4 keypoint
    '''
    confident = None
    boxes = None
    keypoints = None
    for result in inputs:
        confident = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()  # Boxes object for bbox outputs
        keypoints = result.keypoints.xy.cpu().numpy()  # Keypoints object for pose outputs
    if len(confident) == 0:
        return False, None, None, None
    best_idx = np.argmax(confident)
    return True, confident[best_idx], boxes[best_idx], keypoints[best_idx]

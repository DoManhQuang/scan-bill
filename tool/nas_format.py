import json
import os
import numpy as np
data_combined = {"images": {}, "annotations": []}
json_path = r"label"  # Use forward slashes or r prefix for raw string
image_id = 1

for filename in os.listdir(json_path):
    if filename.endswith(".json"):
        with open(os.path.join(json_path, filename), "r") as file:
            json_data = json.load(file)
            image_path = json_data["imagePath"]
            data_combined["images"][str(image_id)] = image_path

            # Process annotations
            annotations = []
            for shape in json_data.get('shapes', []):
                keypoint = shape['points']
                max_x = max([v[0] for v in keypoint])
                max_y = max([v[1] for v in keypoint])
                min_x = min([v[0] for v in keypoint])
                min_y = min([v[1] for v in keypoint])
                
                bbox = [max_x, max_y, min_x, min_y]
                [kpts.append(1) for kpts in keypoint]
                print(">>>>>>> keypts : ", keypoint)
                annotation = {
                    "image_id": image_id,
                    "bbox": bbox,
                    "keypoints": keypoint,
                    "num_keypoints": 4,
                    "category_id": 1
                }
                annotations.append(annotation)

            data_combined["annotations"].extend(annotations)
            image_id += 1

# Write data to a JSON file without line breaks
with open("kp.json", "w") as output_file:
    output_file.write(json.dumps(data_combined, separators=(',', ':')))

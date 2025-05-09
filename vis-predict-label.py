import cv2
import numpy as np
import os
from glob import glob

class_names = [
    'bridge', 'harbor', 'oil_tank',
    'playground', 'airport', 'wind_turbine'
]

colors = [
    (255, 0, 0),   # 蓝色: bridge
    (0, 255, 0),   # 绿色: harbor
    (0, 0, 255),   # 红色: oil_tank
    (255, 255, 0), # 青色: playground
    (255, 0, 255), # 品红: airport
    (0, 255, 255)  # 黄色: wind_turbine
]


def visualize_annotations(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob(os.path.join(image_dir, '*.jpg')) + \
                  glob(os.path.join(image_dir, '*.png'))

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, base_name + '.txt')

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 9:
                continue

            try:
                class_idx = int(parts[0])
                coords = list(map(float, parts[1:]))
            except:
                continue

            if class_idx < 0 or class_idx >= len(class_names):
                continue

            points = []
            for i in range(0, 8, 2):
                x = coords[i] * 512
                y = coords[i + 1] * 512
                x = int(np.clip(round(x), 0, 511))
                y = int(np.clip(round(y), 0, 511))
                points.append((x, y))

            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True,
                          color=colors[class_idx], thickness=2)

        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)


if __name__ == "__main__":
    # vis optical
    visualize_annotations(
        image_dir = 'M4-SAR/optical/test/images',
        label_dir = 'runs/predict/M4-SAR/yolo11-obb-E2E-OSDet-300e/labels',
        output_dir = 'runs/predict/M4-SAR/yolo11-obb-E2E-OSDet-300e/vis/optical'  # need mkdir runs/predict/M4-SAR/yolo11-obb-E2E-OSDet-300e/vis/optical
    )
    # vis sar
    visualize_annotations(
        image_dir = 'M4-SAR/sar/test/images',
        label_dir = 'runs/predict/M4-SAR/yolo11-obb-E2E-OSDet-300e/labels',
        output_dir = 'runs/predict/M4-SAR/yolo11-obb-E2E-OSDet-300e/vis/sar'  # need mkdir runs/predict/M4-SAR/yolo11-obb-E2E-OSDet-300e/vis/optical
    )
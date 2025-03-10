import os
import cv2
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN
import shutil

# Định nghĩa thư mục đầu vào và đầu ra
input_dirs = ["archive/train", "archive/test"]  # Cả train & test
output_dirs = ["processed_data/train", "processed_data/test"]

# Danh sách cảm xúc cần xử lý
emotions = ["angry", "disgust", "fear", "sad"]
for output_dir in output_dirs:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
detector = MTCNN()
def process_images(input_dir, output_dir):
    """Tiền xử lý ảnh từ input_dir và lưu vào output_dir, chỉ cắt mặt (không augmentation)."""
    for emotion in emotions:
        input_path = os.path.join(input_dir, emotion)
        output_path = os.path.join(output_dir, emotion)
        os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(input_path):
            print(f"⚠️ Folder không tồn tại: {input_path}")
            continue
        for img_name in tqdm(os.listdir(input_path), desc=f"Processing {emotion} ({input_dir})"):
            img_path = os.path.join(input_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue  
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(image_rgb)
            if faces:
                x, y, w, h = faces[0]['box']
                face = image_rgb[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))  
                output_face_path = os.path.join(output_path, img_name)
                cv2.imwrite(output_face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
if __name__ == "__main__":
    process_images("archive/train", "processed_data/train")
    process_images("archive/test", "processed_data/test")
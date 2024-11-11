# prepare_data.py

import os
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
from torchvision import transforms
from tqdm import tqdm

DATA_BASE = 'data_base'
DATA_FINE_TUNING = 'data_fine_tuning'

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_image_paths(root_dir):
    image_paths = []
    classes = os.listdir(root_dir)
    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        if os.path.isdir(cls_path):
            for img in os.listdir(cls_path):
                if img.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_paths.append(os.path.join(cls_path, img))
    return image_paths, classes

def find_max_dimensions(image_paths, mtcnn):
    max_width = 0
    max_height = 0
    for img_path in tqdm(image_paths, desc="Finding max dimensions"):
        img = Image.open(img_path).convert('RGB')
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            x1, y1, x2, y2 = boxes[0]
            face = img.crop((x1, y1, x2, y2))
            max_width = max(max_width, face.width)
            max_height = max(max_height, face.height)
    return max_width, max_height

def pad_image(image, target_width, target_height):
    delta_width = target_width - image.width
    delta_height = target_height - image.height
    padding = (0, 0, delta_width, delta_height)
    return ImageOps.expand(image, padding, fill=(0, 0, 0))

def prepare_data():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=0, device=device)

    image_paths, classes = get_image_paths(DATA_BASE)
    create_dir(DATA_FINE_TUNING)
    for cls in classes:
        create_dir(os.path.join(DATA_FINE_TUNING, cls))

    max_width, max_height = find_max_dimensions(image_paths, mtcnn)

    for img_path in tqdm(image_paths, desc="Processing images"):
        cls = os.path.basename(os.path.dirname(img_path))
        img = Image.open(img_path).convert('RGB')
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            x1, y1, x2, y2 = boxes[0]
            face = img.crop((x1, y1, x2, y2))
            face = pad_image(face, max_width, max_height)
            save_path = os.path.join(DATA_FINE_TUNING, cls, os.path.basename(img_path))
            face.save(save_path)
        else:
            print(f"Face not detected in {img_path}")

if __name__ == '__main__':
    prepare_data()

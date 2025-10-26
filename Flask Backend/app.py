import os
from PIL import Image
""" 
base_path = r"E:\internship project\Classification Model\Classification dataset"
folders = ['train\\aadhar', 'train\\non_aadhar', 'val\\aadhar', 'val\\non_aadhar']

for folder in folders:
    path = os.path.join(base_path, folder)
    for i, file in enumerate(os.listdir(path)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path, file)
            img = Image.open(img_path).convert("RGB")  # rgb conversion
            img = img.resize((640, 640))  # resize to 640x640
            new_name = f"{folder.split('\\')[-1]}_{i+1}.jpg"
            img.save(os.path.join(path, new_name))
 """

import cv2, os

img_folder = r"E:\internship project\Detection Model\detection_dataset\images\val"
label_folder = r"E:\internship project\Detection Model\detection_dataset\labels\val"

count = 0  # Counter to track number of images displayed
max_images = 9  # Maximum images to show

for file in os.listdir(img_folder):
    if file.endswith(".jpg"):
        img = cv2.imread(os.path.join(img_folder, file))
        h, w, _ = img.shape
        label_file = os.path.join(label_folder, file.replace(".jpg",".txt"))
        with open(label_file) as f:
            for line in f:
                c, x, y, bw, bh = map(float, line.strip().split())
                x1, y1 = int((x-bw/2)*w), int((y-bh/2)*h)
                x2, y2 = int((x+bw/2)*w), int((y+bh/2)*h)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
        cv2.imshow("Check", img)
        cv2.waitKey(500)

        count += 1
        if count >= max_images:
            break  # Stop after showing 10 images

cv2.destroyAllWindows()
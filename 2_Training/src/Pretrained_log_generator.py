# pretrained_log_generator.py

from keras_yolo3.yolo import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of current script
pretrained_weights_path = os.path.join(BASE_DIR, "keras_yolo3", "yolov3.h5")
# --- Step 1: Specify paths ---
log_dir = "Data/Model_Weights/pretrained_train"

# --- Step 2: Initialize YOLO with pretrained weights ---
yolo = YOLO(
    model_path=pretrained_weights_path,                # No previously trained model
    pretrained_weights=pretrained_weights_path,
    log_dir=log_dir,
    epochs=5,                       # Short training to generate logs
    batch_size=8
)

# --- Step 3: Train on your dataset ---
yolo.train()

print(f"Pretrained YOLO logs saved in: {log_dir}")

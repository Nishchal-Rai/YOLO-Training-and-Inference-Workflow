import os
import subprocess
import time
import argparse
import sys

FLAGS = None

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_folder = os.path.join(root_folder, "2_Training", "src", "keras_yolo3")
data_folder = os.path.join(root_folder, "Data")
model_folder = os.path.join(data_folder, "Model_Weights")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--download_folder",
        type=str,
        default=download_folder,
        help="Folder to download weights to. Default is " + download_folder,
    )

    parser.add_argument(
        "--is_tiny",
        default=False,
        action="store_true",
        help="Use the tiny Yolo version for better performance and less accuracy. Default is False.",
    )

    FLAGS = parser.parse_args()

    if not FLAGS.is_tiny:
        weights_file = "yolov3.weights"
        h5_file = "yolo.h5"
        cfg_file = "yolov3.cfg"
        gdrive_id = "1ENKguLZbkgvM8unU3Hq1BoFzoLeGWvE_"
    else:
        weights_file = "yolov3-tiny.weights"
        h5_file = "yolo-tiny.h5"
        cfg_file = "yolov3-tiny.cfg"
        gdrive_id = "1mIEZthXBcEguMvuVAHKLXQX3mA1oZUuC"

    weights_path = os.path.join(download_folder, weights_file)

    if not os.path.isfile(weights_path):
        print(f"\nDownloading {weights_file} to {download_folder}\n")
        start = time.time()

        # Ensure gdown is installed
        try:
            import gdown
        except ImportError:
            subprocess.call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        # Download inside keras_yolo3 folder
        os.makedirs(download_folder, exist_ok=True)
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, weights_path, quiet=False)

        end = time.time()
        print(f"\n✅ Downloaded {weights_file} in {end - start:.1f} seconds.")
        print(f"Saved to: {weights_path}\n")
    else:
        print(f"✅ {weights_file} already exists in {download_folder} — skipping download.\n")

    # Convert weights to .h5
    print("Converting Darknet weights to Keras model...")
    convert_script = os.path.join(download_folder, "convert.py")

    subprocess.call(
        [sys.executable, convert_script, cfg_file, weights_file, h5_file],
        cwd=download_folder
    )

    print(f"\n Conversion complete! Model saved as {os.path.join(download_folder, h5_file)}")

import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont

# --- Paths ---
csv_path = "Data/Training_Labels/Training_Images_labels.csv"
img_folder = "Data/Source_Images/MyTest_Images/"
output_folder = "Data/Source_Images/Labeled_Images/"

# --- Create output folder if not exists ---
os.makedirs(output_folder, exist_ok=True)

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Loop through each image ---
for img_name in df['image'].unique():
    img_path = os.path.join(img_folder, img_name)
    if not os.path.exists(img_path):
        print(f"⚠️ Image {img_name} not found, skipping...")
        continue

    # Open image
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Get all boxes for that image
    boxes = df[df['image'] == img_name]

    # Draw each bounding box
    for _, row in boxes.iterrows():
        xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
        label = row['label']

        # Draw rectangle in green
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="lime", width=3)

        # Optional: Add label text
        try:
            draw.text((xmin + 5, ymin + 5), label, fill="lime")
        except:
            pass  # In case text drawing fails due to font issues

    # Save the labeled image
    save_path = os.path.join(output_folder, img_name)
    image.save(save_path)

    print(f"✅ Saved labeled image: {save_path}")

print("\n🎉 All labeled images saved in:", output_folder)

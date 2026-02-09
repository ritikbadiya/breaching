import pandas as pd
import io
import os
from PIL import Image
import gc

def extract_and_save_images(df, output_dir):
    # Loop through rows and save images
    for index, row in df.iterrows():
        # HF Parquet files usually store image data in a 'bytes' field 
        # inside an 'image' column or directly as a dictionary
        image_data = row['image']['bytes']

        # Open image from bytes and save
        img = Image.open(io.BytesIO(image_data))

        # Use the provided path/filename if available, else use index
        img_filename = row['image'].get('path', f"image_{index}.jpg")
        class_name = img_filename.split('_')[0]
        img_filename = '_'.join(img_filename.split('_')[1:])  # Get the rest of the filename after class name
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)  # Create class subdirectory
        img.save(os.path.join(output_dir, class_name, img_filename))
        # img.save(os.path.join(output_dir, img_filename))

# Load the parquet file
BASE_DIR = "/home/siladittyamanna/Documents/smanna/iisc/work1/imagenet1k_64x64/data"
parquet_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.parquet')]
test_files = [f for f in parquet_files if 'test' in f]  # Assuming there's a test file
train_files = [f for f in parquet_files if 'train' in f]  # Assuming there's a train file
valid_files = [f for f in parquet_files if 'valid' in f]  # Assuming there's a valid file


# Create an output directory
output_dir = "./dataset/in1k_64x64"
os.makedirs(output_dir, exist_ok=True)

for split in ['train', 'valid', 'test']:
    print(f"Extracting {split} images...")
    if split=='test':
        for tf in test_files:
            df = pd.read_parquet(os.path.join(BASE_DIR, tf))
            extract_and_save_images(df, os.path.join(output_dir, split))
            del df
            gc.collect()  # Free up memory after processing each file
    elif split=='train':
        for tf in train_files:
            df = pd.read_parquet(os.path.join(BASE_DIR, tf))
            extract_and_save_images(df, os.path.join(output_dir, split))
            del df
            gc.collect()  # Free up memory after processing each file
    elif split=='valid':
        for tf in valid_files:
            df = pd.read_parquet(os.path.join(BASE_DIR, tf))
            extract_and_save_images(df, os.path.join(output_dir, split))
            del df
            gc.collect()  # Free up memory after processing each file

print(f"Extraction complete! Images saved to {output_dir}")

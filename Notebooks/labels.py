import os
import pandas as pd


csv_path = r"D:\Object_detection\Data\faces.csv"
images_dir = r"D:\Object_detection\Data\images"
labels_dir = r"D:\Object_detection\Data\labels"

os.makedirs(labels_dir, exist_ok=True)

df = pd.read_csv(csv_path)


for idx, row in df.iterrows():
    img_name = row['image_name']
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    width_img = row['width']
    height_img = row['height']
    
    
    x_center = (x0 + x1) / 2.0 / width_img
    y_center = (y0 + y1) / 2.0 / height_img
    w = (x1 - x0) / width_img
    h = (y1 - y0) / height_img
    
   
    label_file = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
    with open(label_file, 'a') as f:  
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

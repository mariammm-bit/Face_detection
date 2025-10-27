from ultralytics import YOLO
import os
import glob
import cv2

model = YOLO(r"D:\Object_detection\runs\detect\train3\weights\best.pt")

images_path = r"D:\Object_detection\Data\images\*.jpg"
output_dir = r"D:\Object_detection\outputs"
os.makedirs(output_dir, exist_ok=True)

for img_path in glob.glob(images_path):
    results = model(img_path)
    result = results[0]
    plotted_img = result.plot()
    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, plotted_img)

    print(f" Saved: {out_path}, Detected {len(result.boxes)} faces")

print("Detection finished. All results saved in:", output_dir)

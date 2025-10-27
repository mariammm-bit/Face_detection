from ultralytics import YOLO


model = YOLO("yolov8n.pt")


model.train(
    data="dataset.yaml",  
    epochs=10,            
    imgsz=640, 
    batch=4,           
    device="cpu"           
)
                
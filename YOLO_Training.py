from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolo11m.pt")

device = "cpu"  # 'cuda' for GPU, 'cpu' for CPU
model.to(device)

# Train the model on the custom dataset
model.train(data="LPR/data.yaml", epochs=10, device=device)

# Export the trained model to ONNX format
model.export(format="onnx")

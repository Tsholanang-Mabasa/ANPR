from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")

# Ensure the model is using the GPU
device = "cuda"  # 'cuda' for GPU, 'cpu' for CPU
model.to(device)

# Train the model on the custom dataset, specifying the GPU usage
model.train(data="/home/tsholanang/PycharmProjects/COS 731/Water1/data.yaml", epochs=150, device=device)

# Export the trained model to ONNX format
model.export(format="onnx")
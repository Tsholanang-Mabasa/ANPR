from ultralytics import YOLO

model = YOLO("/home/tsholanang/PycharmProjects/COS 731/detect/train_250_epochs/weights/best.pt")
model.export(format="tflite")

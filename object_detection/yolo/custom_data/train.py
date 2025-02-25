from ultralytics import YOLO

if __name__ == '__main__':
    """
    Main entry point for training the YOLO model.
    """
    # Load a model
    model = YOLO("yolo11n.pt")  # build a new model from scratch

    # Use the model
    results = model.train(data=r"E:\GitHub\DeepLearning\apple-palm-1\data.yaml", epochs=10, imgsz=640)  # train the model
    """
    Trains the YOLO model with the specified parameters.
    """

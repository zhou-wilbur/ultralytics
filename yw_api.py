from ultralytics import YOLOWorld


if __name__ == "__main__":
    # Initialize a YOLO-World model
    model = YOLOWorld("yolov8x-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes

    model.set_classes(["cat", "dog"])

    # Execute inference with the YOLOv8s-world model on the specified image
    results = model.predict("test_images/cat.jpg")

    # Show results
    results[0].show()
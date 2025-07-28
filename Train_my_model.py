from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="D:\workspace\ATCC\ATCCdataset\data.yaml",
        epochs=50,
        imgsz=640,
        batch=10,
        name="ATCC_modelV1",
        device="cuda"
    )

if __name__ == "__main__":
    main()

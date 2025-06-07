from ultralytics import YOLO





if __name__ == '__main__':    

    model = YOLO("yolov8m.pt")  # load a pretrained model 
    results = model.train(data="config.yaml", epochs=150, imgsz=512, patience=35,batch=-1,device="0",seed=42,lr0=0.0001,weight_decay=0.0005,name="yolo11n_500px")
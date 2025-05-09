from ultralytics import YOLO

def main():
    data ='ultralytics/cfg/datasets/M4-SAR.yaml'
    cfg = 'ultralytics/cfg/models/benchmark/yolo11-obb-E2E-OSDet.yaml'
    model = YOLO(cfg)
    project = 'runs/train/M4-SAR/'
    name = 'yolo11-obb-E2E-OSDet-300e'
    model.train(data=data, epochs=300, batch=64, imgsz=512, name=name, resume=False, device=[0,1], project=project) # resume train setting 'resume=True' and 'cfg = 'runs/train/M4-SAR/yolo11-obb-E2E-OSDet-300e/weights/last.pt''

if __name__ == '__main__':
    main()

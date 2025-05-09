from ultralytics import YOLO
 
def main():
    model = YOLO('runs/train/M4-SAR/yolo11-obb-E2E-OSDet-300e/weights/best.pt')
    metrics = model.val(split='test', imgsz=512, device=0, batch=1, workers=4, project='runs/test/M4-SAR', name='yolo11-obb-E2E-OSDet-300e')
    map75 = metrics.box.map75
    print(map75)
  
if __name__ == '__main__':
    main()

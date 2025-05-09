from ultralytics import YOLO
 
def main():
    model = YOLO('runs/train/M4-SAR/yolo11-obb-E2E-OSDet-300e/weights/best.pt')
    batch = 1 # batch only set 1
    source = 'M4-SAR/detect/' # M4-SAR/detect/optical/....jpg  and M4-SAR/detect/sar/....jpg
    project = 'runs/predict/' 
    name = 'M4-SAR/yolo11-obb-E2E-OSDet-300e'
    model(source=source, save_conf=True, imgsz=512, save=False, device=0, batch=batch, workers=1, save_txt=True, project=project, name=name)

if __name__ == '__main__':
    main()

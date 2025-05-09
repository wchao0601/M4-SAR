<div align="center">
<!-- <h1> M4-SAR </h1> -->
<h3> <a href="url">M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection</h3>
<h4> arXiv 2025</h4>
</div>

## **Examples of scenes and categories in the proposed M4-SAR dataset, along with the instance size and aspect ratio distributions.**
![image](https://github.com/wchao0601/M4-SAR/blob/master/img/motivation.png)

## **Statistical visualization of category attributes in the proposed M4-SAR dataset.**
![image](https://github.com/wchao0601/M4-SAR/blob/master/img/data-statistics.png)

## **Overall Framework.**
![image](https://github.com/wchao0601/M4-SAR/blob/master/img/overall-network.png)

## **Architectural details of the proposed FAM, CMIM, and AFM modules.**
![image](https://github.com/wchao0601/M4-SAR/blob/master/img/FAM-CMIM-AFM.png)

## M4-SAR Dataset
| Dataset | Download Link | Code |
| --- | --- | --- |
| MS-SAR | [Download](https://kaggle.com/datasets/a8ca500cbad658d8ae1af3d1f84566a5b4e94fe0ddb0be801c9e2f672db36a57)| 0601 |


## Dataset and Label Structure
![image](https://github.com/wchao0601/M4-SAR/blob/master/img/m4-sar-structure.png)

See [Segmentation Docs](https://docs.ultralytics.com/tasks/segment/) for usage examples with these models trained on [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/), which include 80 pre-trained classes.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) | 640                   | 38.9                 | 32.0                  | 65.9 ± 1.1                     | 1.8 ± 0.0                           | 2.9                | 10.4              |
| [YOLO11s-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) | 640                   | 46.6                 | 37.8                  | 117.6 ± 4.9                    | 2.9 ± 0.0                           | 10.1               | 35.5              |
| [YOLO11m-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) | 640                   | 51.5                 | 41.5                  | 281.6 ± 1.2                    | 6.3 ± 0.1                           | 22.4               | 123.3             |
| [YOLO11l-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) | 640                   | 53.4                 | 42.9                  | 344.2 ± 3.2                    | 7.8 ± 0.2                           | 27.6               | 142.2             |
| [YOLO11x-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) | 640                   | 54.7                 | 43.8                  | 664.5 ± 3.2                    | 15.8 ± 0.7                          | 62.1               | 319.0             |



## Acknowledgment
- This repo is based on [Ultralytics](https://github.com/ultralytics/ultralytics), [CFT](https://github.com/DocF/multispectral-object-detection), [CLANet](https://github.com/hexiao0275/CALNet-Dronevehicle), [CSSA](https://github.com/artrela/mulitmodal-cssa), [CMADet](https://github.com/VDT-2048/DVTOD), [ICAFusion](https://github.com/chanchanchan97/ICAFusion) and [MMIDet](https://github.com/joewybean/MMI-Det) which are excellent works.
- We thank the [STTrack](https://github.com/NJU-PCALab/STTrack) and [YOLOv12](https://github.com/sunsmarterjie/yolov12) libraries, which help us to implement our ideas quickly.

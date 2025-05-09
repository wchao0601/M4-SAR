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


| Model   | size (pixels) | #P(M) | Tra.T (h) | Inf.T (ms) | AP50 | AP75 | mAP |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [CFT](https://github.com/DocF/multispectral-object-detection) | 512 x 512       | 53.8     | 32.0       | 40.6   | 84.6  | 68.9  | 59.9   |
| [CLANet](https://github.com/hexiao0275/CALNet-Dronevehicle)   | 512 x 512       | 48.2     | 37.8       | 29.1   | 84.6  | 68.5  | 59.6   |
| [CSSA](https://github.com/artrela/mulitmodal-cssa)            | 512 x 512       | 13.5     | 41.5       | 12.3   | 83.4  | 66.4  | 58.0   |
| [CMADet](https://github.com/VDT-2048/DVTOD)                   | 512 x 512       | 41.5     | 42.9       | 46.7   | 81.5  | 63.5  | 55.7   |
| [ICAFusion](https://github.com/chanchanchan97/ICAFusion)      | 512 x 512       | 29.0     | 43.8       | 23.6   | 84.5  | 67.3  | 58.8   |
| [MMIDet](https://github.com/joewybean/MMI-Det)                | 512 x 512       | 53.8     | 43.8       | 41.9   | 84.8  | 68.6  | 59.8   |
| [E2E-OSDet](https://github.com/wchao0601/M4-SAR)              | 512 x 512       | 27.5     | 43.8       | 20.9   | 85.7  | 70.3  | 61.4   |



## Acknowledgment
- This repo is based on [Ultralytics](https://github.com/ultralytics/ultralytics), [CFT](https://github.com/DocF/multispectral-object-detection), [CLANet](https://github.com/hexiao0275/CALNet-Dronevehicle), [CSSA](https://github.com/artrela/mulitmodal-cssa), [CMADet](https://github.com/VDT-2048/DVTOD), [ICAFusion](https://github.com/chanchanchan97/ICAFusion) and [MMIDet](https://github.com/joewybean/MMI-Det) which are excellent works.
- We thank the [STTrack](https://github.com/NJU-PCALab/STTrack) and [YOLOv12](https://github.com/sunsmarterjie/yolov12) libraries, which help us to implement our ideas quickly.

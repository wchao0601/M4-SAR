<div align="center">
<!-- <h1> M4-SAR </h1> -->
<h3> <a href="https://arxiv.org/abs/2505.10931">M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection</h3>
<h4> 2025</h4>
</div>

## **Examples of scenes and categories in the proposed M4-SAR dataset.**
<p align="center"> <img src="https://github.com/wchao0601/M4-SAR/blob/master/img/motivation.png" width="90%"> </p>

## **Statistical visualization of category attributes in M4-SAR dataset.**
<p align="center"> <img src="https://github.com/wchao0601/M4-SAR/blob/master/img/data-statistics.png" width="90%"> </p>

## **Overall Framework.**
<p align="center"> <img src="https://github.com/wchao0601/M4-SAR/blob/master/img/overall-network.png" width="90%"> </p>

## **Architectural details of the proposed FAM, CMIM, and AFM modules.**
<p align="center"> <img src="https://github.com/wchao0601/M4-SAR/blob/master/img/FAM-CMIM-AFM.png" width="90%"> </p>

## Usage
### Installation
Create and activate a conda environment:
```
conda create -n e2e-osdet python=3.11
conda activate e2e-osdet
```
Install the required packages:
```
git clone https://github.com/wchao0601/M4-SAR.git
cd M4-SAR/
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install seaborn thop timm einops
cd STTrack/mamba_install/causal-conv1d
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ../selective_scan
pip install .
cd M4-SAR/ 
pip install -r requirements.txt
```
We provide a complete environment configuration [log](https://github.com/wchao0601/M4-SAR/blob/master/create-environment-log.docx) for your reference.

### Optical and SAR Attributes by File Range

| File Range        | Optical Image Resolution | SAR Image Polarization |
|:---:|:---:|:---:|
| 1.jpg ~ 56087.jpg      | 10 meters                | VH                     |
| 56088.jpg ~ 112174.jpg | 60 meters           | VV                     |


### Data Preparation

| Dataset | Link1 | Link2 | Link3 | SR & Pola. | Image Size | Category | Ins.num | Img.num |
| :---: | :---: | :---: | :---: | :---:| :---: | :---: | :---: | :---: |
| M4-SAR | [Kaggle](https://kaggle.com/datasets/a8ca500cbad658d8ae1af3d1f84566a5b4e94fe0ddb0be801c9e2f672db36a57)|[Baidu](https://pan.baidu.com/s/14iuaf_2ymzpP68EJY0dUyg?pwd=0601)|[Hug-Face](https://huggingface.co/datasets/wchao0601/m4-sar)|10M, 60M, VH, VV|512 x 512|6|981,862|112,174|

### Dataset and Label Structure
<p align="center"> <img src="https://github.com/wchao0601/M4-SAR/blob/master/img/m4-sar-structure.png" width="90%"> </p>


### Single-GPU Train
```python
# please set 'device=0' in train.py
python train.py
```

### Multi-GPU Train
```python
# please set 'device=[0,1]' in train.py
python multigpu-train.py
```

### Test
```python
python test.py
```

### Gen-Predict
```python
python gen-predict-label.py
```

### Vis-Predict
```python
python vis-predict-label.py
```

### Gen-Heatmap
```python
python gen-heatmap.py
```

## Results

|  Model     |Weight Link |  Img size (pixels)  |  #Para(M)  |  Tra.Time (h)  |  Inf.Time (ms)  |  AP50 (%)  |  AP75 (%)  |  mAP (%)  |
| :---:      | :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: |
|  CFT       |[Download](https://pan.baidu.com/s/1KjlbzaW_KcsyKyQ7ziUwDg?pwd=0601)    |  512 x 512  |  53.8  |  60.6  |  40.6    |  84.6   |  68.9   |  59.9    |
|  CLANet    |[Download](https://pan.baidu.com/s/1xq7p5ujbRh86WaoxVnEIag?pwd=0601)    |  512 x 512  |  48.2  |  56.2  |  29.1    |  84.6   |  68.5   |  59.6    |
|  CSSA      |[Download](https://pan.baidu.com/s/1M8atC_WC5IUsBEfoQanJ2g?pwd=0601)    |  512 x 512  |  13.5  |  25.7  |  12.3    |  83.4   |  66.4   |  58.0    |
|  CMADet    |[Download](https://pan.baidu.com/s/1pnZoEzIbf9Z5KQnQbN4vXg?pwd=0601)    |  512 x 512  |  41.5  |  52.4  |  46.7    |  81.5   |  63.5   |  55.7    |
|  ICAFusion |[Download](https://pan.baidu.com/s/186bPEbk_BwvUXkZD_M1Y7Q?pwd=0601)    |  512 x 512  |  29.0  |  47.7  |  23.6    |  84.5   |  67.3   |  58.8    |
|  MMIDet    |[Download](https://pan.baidu.com/s/1iB3x_cmOHJFmSVB2zSUsBw?pwd=0601)    |  512 x 512  |  53.8  |  49.9  |  41.9    |  84.8   |  68.6   |  59.8    |
|  E2E-OSDet |[Download](https://pan.baidu.com/s/1GFUONCYPBntRg5_IpUqRYg?pwd=0601)    |  512 x 512  |  27.5  |  42.1  |  20.9    |  85.7   |  70.3   |  61.4    |

### All weights
[Google Drive](https://drive.google.com/file/d/1ZOGOBLtZEg1pQ_0SkqclgP5XXJsYUkU1/view?usp=sharing)

## Contact
If you have any questions, please feel free to contact me via email at wchao0601@163.com

## Citation
If our work is helpful, you can cite our paper:
```
@article{wang2025m4,
  title={M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection},
  author={Wang, Chao and Lu, Wei and Li, Xiang and Yang, Jian and Luo, Lei},
  journal={arXiv preprint arXiv:2505.10931},
  year={2025}
}

```
## Acknowledgment
- This repo is based on [Ultralytics](https://github.com/ultralytics/ultralytics), [CFT](https://github.com/DocF/multispectral-object-detection), [CLANet](https://github.com/hexiao0275/CALNet-Dronevehicle), [CSSA](https://github.com/artrela/mulitmodal-cssa), [CMADet](https://github.com/VDT-2048/DVTOD), [ICAFusion](https://github.com/chanchanchan97/ICAFusion) and [MMIDet](https://github.com/joewybean/MMI-Det) which are excellent works.
- We thank the [STTrack](https://github.com/NJU-PCALab/STTrack) and [YOLOv12](https://github.com/sunsmarterjie/yolov12) libraries, which help us to implement our ideas quickly.

Here I have discussed the setup of my edge device for deployment of YOLOX models
I am using a Jetson Orin Nano super dev kit running Jetpack 6.2.1- cuda 12.6
1) OpenCV
```bash
pip install numpy==1.26.4
pip install opencv-python==4.10.0.82
```
2) Pytorch
Install pytorch for jetpack version compatible and it should be CUDA compatible
Pytorch should run with cuda not just cpu
```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```
Test CUDA compatibility for pytorch
```python
import torch
print(torch.cuda.is_available())
```

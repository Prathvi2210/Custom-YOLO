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
PyTorch is the deep learning framework (whole ecosystem)
torch is the main library that implements: tensors(like NumPy but faster), GPU acceleration, Neural network ops, autograd(backprop) 
```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```
Test CUDA compatibility for pytorch
```python
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0))
```
It should show CUDA available: True
CUDA device: Orin Nano

If you get ouput:
```code
TOrch version: x.x.x+cpu
CUDA available: False
AssertionError: Torch not compiled with CUDA enabled
```
This means PyTorch came from PyPi(CPU build), it has no CUDA support and cant detect GPU.
Remove this and install CUDA supported version
```bash
pip3 uninstall -y torch torchvision torchaudio
```
Verify it is gone with import torch. It should throw error: No module named 'torch'
Then install the correct versions. Sometimes pip ignores NVIDIA's wheel and fall back to PyPI unless you force it not to.
There are two versioning systems involved here:
  a) PyTorch framework version(seen in torch.__version__)
  ex: 2.0.1, 2.2.1
  This tells the API level and PyTorch features
  b) NVIDIA Jetson PyTorch build version(often seen in documentations and forums)
  ex: 23.06, 24.02
  This corresponds to NVIDIA's monthly release, CUDA version, cuDNN version, TensorRT and cusparselt requirements

On Jetpack 6, python3-pytorch is not provided as an apt package. So, sudo apt-get install -y python3-pytorch won't work.
Typically torch for GPU is installed via .whl(wheel) files as seen in the above command.
Here cp310 indicates Python 3.10
and nv24.xx indicated jetpack 6.x comatiblity, nv23 are for jetpack 5.x
the wheel should be supported for the python version on your platform. NVIDIA hosts wheels under: https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/

After the torch is installed and compiled with CUDA, clone YOLOX install it and then run the inference code

Another error I faced while installing torch for cuda was with cuDNN.
```code
ImportError: libcudnn.so.8: cannot open shared object file
```
PyTorch cannot load CUDA kernels without cuDNN. Jetpack 6 ships cuDNN 9 not 8. Any PyTorch wheel linked against cuDNN 8 will not work.

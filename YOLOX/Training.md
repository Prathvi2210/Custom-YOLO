This project was done in 2026, where the google colab was defaulted to python 3.12.x version and did not allow being forced to 3.10.x
YOLOX training on google colab caused version compatibility issues which are documented below along with their practical solutions

1) Setup Environment
```bash
!nvidia-smi
!pip install -U torch torchvision torchaudio
!pip install roboflow loguru tabulate
```
2) Add your dataset either upload the zip files or folders or run the code snippet from roboflow in the correct format
   here I had added the dataset in yolov5 format because apparently it was closest to yolox working, turned out wrong later

3) Clone YOLOX
   Here I am using the default original repo provided by Megvii
```bash
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
%cd YOLOX
!pip install -r requirements.txt
!pip install -v -e .
```
This is the basic YOLOX setup
Hit with the compatibility error for onnx-simplifier==0.4.10. This is written in the requirements.txt in YOLOX repo
YOLOX is not python 3.12 ready so we get the packaging.version.InvalidVersion: 'unknown' error
This is a known breakage for Python>=3.12

Forcing python 3.10 in colab doesn't work, colab doesn't actually let you switch the system python anymore. The commands will look like they worked but the kernels will stay pinned to 3.12

Only solution here is to avoid installing the onnx-simplifier to train
```bash
!pip install -e . --no-deps #install only yolox no dependencies
```
This also wont work because with python 3.12: pip 24.x, YOLOX uses legacy setup.py
pip is trying PEP-660 editable builds and YOLOX does not support PEP-660
```bash
python setup.py develop #Installing YOLOX in legacy editable mode
```
In modern pip tools, this is also internally calling 
```bash
pip install -e . --use-pep517
```
Install only training-safe dependencies: 
```bash
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
%cd YOLOX
!pip install -U pip setuptools wheel
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install opencv-python loguru tqdm tabulate psutil tensorboard pycocotools thop
```

4) Create custom YOLOX experiment file. The file provided in the repo is in the exps/example/default directory and it is for default dataset
   Need to create one for custom dataset training: exps/example/custom/custom_yolox_s.py
```bash
%%write exps/example/custom/custom_yolox_s.py
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.depth = 0.33
        self.width = 0.5 #these 2 terms are what decide the parameters. i.e 0.33 and 0.5 means it is small model, 0.67 and 0.75 means it is medium model
        self.input_size = (832x 832) #default yolo performance is on 640, but my camera was high definition so I trained in 832
        self.random_size = (14, 26)
        self.max_epoch = 100 #first test with 5 epochs if working properly set the final value
        self.data_num_workers = 4
        self.eval_interval= 10 #set this according to the max_epoch value. It should be around 10% REASON HERE
```

5) Training command
```bash
!python tools/train.py \
  -f exps/example/custom/custom_yolox_s.py \
  -d 1 \
  -b 16 \
  --fp16 \
  -o \
  -c yolox_s.pth  \
  --data ../yolox_data.yaml
```
This setup is meant to: use pretrained YOLOX_s weights, mixed precision enabled, optimized training defaults

6) Evaluate
```bash
!python tools/eval.py \
  -f exps/example/custom/custom_yolox_s.py \
  -c YOLOX_outputs/custom_yolox_s/best_ckpt.pth \
  -b 8 \
  -d 1 \
  --conf 0.01


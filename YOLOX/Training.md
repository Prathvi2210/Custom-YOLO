This project was done in early 2026, where the google colab was defaulted to python 3.12.x version and did not allow being forced to 3.10.x
Current Colab GPUs use CUDA 12.x. Older versions and their wheels are incompatible
YOLOX training on google colab caused version compatibility issues which are documented below along with their practical solutions

1) Setup Environment
```bash
!nvidia-smi
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU OK:", torch.cuda.is_available())
```
Don't in colab:
```
!pip install -U torch torchvision torchaudio
!pip install roboflow loguru tabulate
```
2) Add your dataset either upload the zip files or folders or run the code snippet from roboflow in the correct format.
   here I had added the dataset in yolov5 format because apparently it was closest to yolox working, turned out wrong later.
   YOLOX needs extra conversion setup to train on yolo format dataset
   Pascal VOC format is older preference
   I have used COCO-JSON format. It expects COCO-style annotations internally (optional, not a preference)

   Roboflow's current COCO export doesn't always create an annotations/ folder. Instead the annotations.json files are saved in the respective train/, valid/ and test/ folders along with the images. We need to point the dataset paths carefully.
   Verify the number of classes in the json files, that number is needed for training:
   ```bash
   import json
   with open("/content/dataset/train/_annotations.coco.json") as f:
       data = json.load(f)
   print("Num classes:", len(data["categories"]))
   print("classes:", [c["name"] for c in data["categories"]])
   ```
   The json file can be opened on system or browser and it can directly be checked without any code
   Now we need to restructure the dataset: YOLOX always constructs the annotation path internally as: <dataset_directory>/annotations/instances_train2017.json
   Same for val2017.json
   And the images folders are to be renamed to: train2017 and val2017 too
   JSON annotations have each image entry as train2017/xxxxx but roboflow physically stores in train/xxxxx. They are out of sync
4) Clone YOLOX
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
YOLOX's requirements.txt is NOT a 'runtime requirements' file. It is a 'full feature + export + legacy dev' requirements file
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

Instead final working commands on colab:
```bash
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
%cd YOLOX
#install dependencies
!pip install -U pip
!sed '/onnx-simplifier/d' requirements.txt > requirements_train.txt
!pip install -r requirements_train.txt
#Editable install
!pip instaall -v -e . --no-build-isolation --no-deps
```
here even though onnx-simplifier is disabled from requirements.txt, when we run the editable install, pip drags onnx-simplifier back to install because YOLOX hardcodes dependencies in TWO places: requirements.txt and setup.py/pyproject.toml.
Inside YOLOX'x packaging metadata it says to install onnx and onnx-simplifer. hence we have to install YOLO without its declared dependencies(--no-deps)
verify installation
```bash
import yolox
from yolo.exp import get_exp
print("YOLOX import OK")
```
4) Create custom YOLOX experiment file. 
   Need to create one for custom dataset training: exps/example/custom/yolox_s_rf.py
```bash
cp exps/example/custom/yolox_s.py exps/example/custom/yolox_s_rf.py
```
Note: In the cloned repo only the yolox_s.py is available, no yolox_m.py or yolox_l.py. This is expected in some versions.
So for training a medium or large model, the python file is to be derived. changing the self.depth and self.width to appropriate values.(0.33, 0.5) for small model, (0.67, 0.75) for medium model.
Open that file and make the following changes
```bash
self.data_dir = "/content/dataset"

self.train_ann = "instances_train2017.json"
self.val_ann = "instances_val2017.json"

self.num_classes = <number confirmed in step 2>
self.input_size = (H,W) #YOLO uses height, width format, roboflow uses width, height format
self.test_size = (H,W)

self.max_epoch = 300
self.warmup_epochs = 5 #Gradually increases learning rate to stabilize
self.no_aug_epochs = 15 #Last epochs without mosaic/mixup

self.data_num_workers = 4
self.eval_interval = 5

#Augmentation
self.mosaic_prob = 0.8
self.mixup_prob = 0.1
```
Initially keep epochs at min(1 to 5) to do a dry run before running full training

5) Training command
```bash
!python tools/train.py \
  -f exps/example/custom/custom_yolox_s.py \
  -d 1 \
  -b 16 \
  --fp16 \
  -o 
```
To train with pretrained model weights, faster and better accuracy, just add the standard yolox.pth file to the training command with '-c' prefix. This prefix indicates to 'resume from an existing file'. 
Initially try to train fron scratch for few dry run epochs. If it works then fine tune from pretrained weights
YOLOX training saves results to YOLOX_outputs/yolox_s_rf
To resume interupted training just add the line --resume to the training command

Bigger more complex models use more memory so reduce the batch size when training them

YOLOX does not accept --data
That flag exists in v5,v8 etc. In YOLOX the dataset paths are defined inside the experiment file, not passed via command line
Same as --max_epoch command. It should be in the exp file.

6) Evaluate
```bash
!python tools/eval.py \
  -f exps/example/custom/custom_yolox_s.py \
  -c YOLOX_outputs/custom_yolox_s/best_ckpt.pth \
  -b 8 \
  -d 1 \
  --conf 0.01
```
Golden rules:
Never uninstall torch in cuda
Never pin torch/ CUDA
Never install YOLOX before torch
Use official Megvii YOLOX repo
Use colab's default PyTorch
Editable install after deps

Training flags:
Early epochs (15-20) will have:
   high conf_loss
   noisy cls_loss
   near zero AP
   First few epochs will run without mosaic while buffers initialize
Mid Epochs :
   AP rising
   cls_loss separating clearly
   stable validation loss
End (last 15-20) epochs:
   Mosaic and mixup (native to YOLOX) disabled immediately
   AP improvement noticeable with the above point


   

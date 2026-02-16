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
4) Create custom YOLOX experiment file. The file provided in the repo is in the exps/example/default directory and it is for default dataset
5) Need to create one for custom dataset training: exps/example/custom/custom_yolox_s.py
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
        self.eval_interval= 5
```

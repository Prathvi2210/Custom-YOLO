YOLOX outputs weights in .pth format
Traditional yolo models have .pt files which are built as an end-to-end product with training, inference, export all wrapped in one. It contains the model weights, architecture metadata, pre/post processing assumptions which makes deployment for inference very easy.
The .pth file is only raw PyTorch weights, not packaged deployable file. Here we control the pre/post processing, model architecture, custom NMS and edge optimization
The repo does come with an official demo for initial use (tools/demo.py): download the .pth file and custom exps file (yolox_s_rf.py) locally
Simple Inference command:
```bash
python tools/demo.py image \
  -f yolox_s_rf.py \
  -c best_ckpt.pth \
  --conf 0.25 \
  --nms 0.65 \
  --tsize 832 \
  --path your_image.jpg \
  --device gpu
```
Preferred option is to export it to ONNX framework and create .onnx file
This is the code snipped for converting .pth to .onnx on google drive
```bash
python tools/export_onnx.py \
  -f exps/example/custom/yolox-s_rf.py \
  -c YOLOX_outputs/yolox_s_rf/best_ckpt.pth \
  --output-name yolox_m.onnx \
  --input-size 832 832 \
  --opset 11
```
This can then be deployed using:
ONNX runtime (CPU/CUDA: widely compatible)
TensorRT (NVIDIA devices optimized)
OpenVINO (Intel devices optimized)

ONNX conversion if preferred on local device if possible for which the theory and code is given below.

TensorRT is the hardware-optimized format for jetson orin's GPU + Tensor cores, working on low latency and low power
For best performance it is recommended that the computer vision models are used in tensorRT format
Specifically the model file will look like TensorRTfp16.engine

Generally for custom training (especially for yolo models) the output of best model weights is saved as a .pt or .pth file.
hence it was of importance that the file should be converted be in .engine format
.onnx is the universal framework language required for conversion
So export the trained weights file as .onnx and then convert that into .engine file

The commands are compatible to TensorRT version 10.7 and pytorch version 2.5.x
It is essential that the pytorch build is with cuda and doesn't fall back to cpu based

The command to convert .pth to .onnx:
```bash
python3 tools/export_onnx.py \
    -f exps/default/yolox_s.py \
    -c yolox_s.pth \
    --output-name yolox_s.onnx \
    --opset 11 \
    --batch-size 1 \
    --decode_in_inference
```
When using a custom trained model, change the exp file. In directory YOLOX/exps/custom/yolox_custom.py
This file should contain the custom training parameters

The command to locally convert the onnx framework into tensorRT:
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolox_s_standard.onnx \
    --saveEngine=yolox_s_standard.engine \
    --fp16 \
    --memPoolSize=workspace:4096M \
    --builderOptimizationLevel=5 \
    --useCudaGraph \
```
Jetson orin nano is capable of a 6Gb:6144M workspace, else workspace:4096M
This will take 10-12 minutes

NOTE: Everytime the software stack is updated i.e. jetpack update of tensorRT version change the conversions need to be done again. An engine file exported in older version is not forward compatible even for offline use

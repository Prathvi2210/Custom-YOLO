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

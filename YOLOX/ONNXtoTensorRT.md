TensorRT is the hardware-optimized format for jetson orin's GPU + Tensor cores, working on low latency and low power
For best performance it is recommended that the computer vision models are used in tensorRT format
Specifically the model file will look like TensorRTfp16.engine

Generally for custom training (especially for yolo models) the output of best model weights is saved as a .pt or .pth file.
Other than being not optimized for deployment on nvidia jetson orin nano device the new coming software stacks (eg. jetpack 6.2.1) are incompatible as they did not have a 'cuda deployed' pytorch version available
this issue will be documented further.
hence it was of importance that the file be in .engine format
.onnx is the universal framework required for conversion
So export the trained weights file as .onnx and then convert that into .engine file

The trained model output will consist of 2 files of the form:
yolox_s_custom.onnx
yolox_s_custom.onnx.data

The command to locally convert the onnx framework into tensorRT:
```bash
/usr/src/tensorrt/bin/trtexec \
--onnx=yolox_s_custom.onnx \
--saveEngine=yolox_s_fp16.engine \
--fp16 \
--memPoolSize=workspace:4096
```
This will take 5-6 minutes

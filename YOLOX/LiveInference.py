import PySpin
import cv2
import torch
import numpy as np

from yolox.exp import get_exp
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
from yolox.utils.visualize import vis

# ----------------------------
# CONFIG
# ----------------------------
EXP_FILE = "exps/example/custom/yolox_s_rf.py"
CKPT_FILE = "best_ckpt.pth"

CONF_THRES = 0.25
NMS_THRES = 0.65
INPUT_SIZE = (832, 832)

# ----------------------------
# LOAD YOLOX MODEL
# ----------------------------
exp = get_exp(EXP_FILE, None)
model = exp.get_model()
model.eval()

ckpt = torch.load(CKPT_FILE, map_location="cuda")
model.load_state_dict(ckpt["model"])
model.cuda()

preproc = ValTransform()

# ----------------------------
# INIT PYSPIN CAMERA
# ----------------------------
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

if cam_list.GetSize() == 0:
    cam_list.Clear()
    system.ReleaseInstance()
    raise RuntimeError("No Teledyne BFS camera detected")

cam = cam_list[0]
cam.Init()

# Set pixel format (VERY IMPORTANT)
cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)

cam.BeginAcquisition()

print("Teledyne BFS camera streaming...")

# ----------------------------
# LIVE LOOP
# ----------------------------
try:
    while True:
        image = cam.GetNextImage(1000)
        if image.IsIncomplete():
            image.Release()
            continue

        frame = image.GetNDArray()  # numpy array (H,W,3)
        image.Release()

        # YOLOX preprocessing
        img, ratio = preproc(frame, None, INPUT_SIZE)
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(
                outputs,
                exp.num_classes,
                CONF_THRES,
                NMS_THRES
            )

        if outputs[0] is not None:
            frame = vis(
                frame,
                outputs[0],
                CONF_THRES,
                exp.class_names
            )

        cv2.imshow("YOLOX Helmet Detection (Teledyne BFS)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.EndAcquisition()
    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
    cv2.destroyAllWindows()

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys, os, getopt
# from tkinter.messagebox import NO
import time

from realsense_pipeline import realsense_pipeline

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


file_path = os.path.dirname(os.path.abspath(__file__))
hrnet_path = os.path.join(file_path, "pretrained_hrnet.pth")  # args["yolo"]

sys.path.append(hrnet_path)
sys.path.append(file_path)

import cv2
import torch
import numpy as np
import torch

from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
# from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_one_box
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from tqdm import tqdm

g_wb = np.array([
    [0, 0., 1, 0],
    [-1, 0, 0., 0.],
    [0, -1, 0, 0.44],
    [0, 0, 0, 1]
])

g_bh = np.array([
    [0.9615, -0.0264, -0.2725, 0.9559],
    [0.0594, 0.9910, 0.1144, -1.1018],
    [0.2675, -0.1264, 0.9544, -2.4053],
    [0, 0, 0, 1]
])
focal_len = 720

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_pose_from_model_output(output, bbox):
    '''
    Compute pose from model prediction
    '''
    pred_cam_root = output.cam_root.squeeze(dim=0).detach().cpu().numpy()
    pred_xyz_jts_29 = output.pred_xyz_jts_29.reshape(29, 3).cpu().data.numpy()
    pred_xyz_jts_29 *= 2.2   # jts is normalized, so multiply 2.2 to get it back
    bbox_xywh = xyxy2xywh(bbox)
    focal = 1000
    focal = focal / 256 * bbox_xywh[2]
    pred_cam_root[2] *= (focal_len / focal)  # 590 is real camera focal length
    jts_in_camera = pred_cam_root[np.newaxis, :] + pred_xyz_jts_29
    # g_bh obtained from camera calibration, human camera in ball camera
    # g_wb, ball camera in world frame

    g_wh = g_wb @ g_bh
    jts_p = np.append(jts_in_camera, np.ones((29, 1)), axis=1)
    jts_in_world = jts_p @ g_wh.T
    jts_in_world = jts_in_world[:, :3]
    return jts_in_world


class PoseEstimate:
    def __init__(self):
        self.device = torch.device('cuda:0')

        # cfg_file = os.path.join(file_path, '256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml')
        # CKPT = os.path.join(file_path, "pretrained_hrnet.pth")

        cfg_file = os.path.join(file_path, '256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d.yaml')
        CKPT = os.path.join(file_path, "pretrained_w_cam.pth")
        cfg = update_config(cfg_file)

        bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
        dummpy_set = edict({
            'joint_pairs_17': None,
            'joint_pairs_24': None,
            'joint_pairs_29': None,
            'bbox_3d_shape': bbox_3d_shape
        })

        self.transformation = SimpleTransform3DSMPLCam(
            dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
            color_factor=cfg.DATASET.COLOR_FACTOR,
            occlusion=cfg.DATASET.OCCLUSION,
            input_size=cfg.MODEL.IMAGE_SIZE,
            output_size=cfg.MODEL.HEATMAP_SIZE,
            depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape=bbox_3d_shape,
            rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
            train=False, add_dpg=False,
            loss_type=cfg.LOSS['TYPE'])

        self.det_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.hybrik_model = builder.build_sppe(cfg.MODEL)

        save_dict = torch.load(CKPT, map_location='cpu')
        if type(save_dict) == dict:
            model_dict = save_dict['model']
            self.hybrik_model.load_state_dict(model_dict)
        else:
            self.hybrik_model.load_state_dict(save_dict)

        self.det_model.cuda()
        self.hybrik_model.cuda()
        self.det_model.eval()
        self.hybrik_model.eval()

        self.det_transform = T.Compose([T.ToTensor()])
        self.global_pos = None

    def detection(self, input_image):

        det_input = self.det_transform(input_image).to(self.device)
        det_output_ = self.det_model([det_input])

        det_output = det_output_[0]
        # selected = det_output_[0]['labels'] == 1
        # det_output = {'boxes': det_output['boxes'][selected], 'labels': det_output['labels'][selected], 'scores': det_output['scores'][selected]}
        tight_bbox = get_one_box(det_output)  # xyxy

        if tight_bbox is not None:
            # Run HybrIK
            # bbox: [x1, y1, x2, y2]
            pose_input, bbox, img_center = self.transformation.test_transform(
                input_image, tight_bbox)

            pose_input = pose_input.to(self.device)[None, :, :, :]
            pose_output = self.hybrik_model(
                pose_input, flip_test=False,
                bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
            )

            jts_in_world = get_pose_from_model_output(pose_output, bbox)
        else:
            jts_in_world = None

        return jts_in_world

    def prepare_detection_window(self):
        self.pipeline = realsense_pipeline()
        # cv2.namedWindow('RealSenseRGB', cv2.WINDOW_AUTOSIZE)
        torch.cuda.empty_cache()

    def run_detection_once(self):
        # time_hist.append(t1)
        t1 = time.perf_counter()
        image = self.pipeline.get_rgb_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t2 = time.perf_counter()
        print("get image delay", (t1 - t2) * 1000)

        global_pos = self.detection(image)  ##检测

        if global_pos is not None:
            self.global_pos = (global_pos, t2)
        t2 = time.perf_counter()
        print("get detection", (t1 - t2) * 1000)
        print(f'Freq ({1/(t2 - t1):.3f} fps)')

    def run_detection_forever(self):
        while True:
            print('detecting')
            self.run_detection_once()

    def get_global_pos(self):
        return self.global_pos

if __name__ == '__main__':
    pose_estimate = PoseEstimate()
    img_path = 'pose.png'
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    for i in range(1000):
        print(i)
        output = pose_estimate.detection(image)
    print(output)
    print(output.shape)

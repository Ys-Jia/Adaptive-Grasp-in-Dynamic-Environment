import os
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
import random
import numpy as np
from pyobb.obb import OBB
from seq2seq import Encoder, Decoder

from pytorch3d.transforms import matrix_to_quaternion

def cam_view2pose(cam_view_matrix):
    """
    In:
        cam_view_matrix: a list of 16 floats, representing a 4x4 matrix.
    Out:
        cam_pose_matrix: Numpy array [4, 4].
    Purpose:
        Convert camera view matrix to pose matrix.
    """
    cam_pose_matrix = np.linalg.inv(np.array(cam_view_matrix).reshape(4, 4).T)
    cam_pose_matrix[:, 1:3] = -cam_pose_matrix[:, 1:3]
    return cam_pose_matrix

def transform_point3s(t, ps):
    """
    In:
        t: Numpy array [4, 4] to represent a transform
        ps: point3s represented as a numpy array [Nx3], where each row is a point.
    Out:
        Transformed point3s as a numpy array [Nx3].
    Purpose:
        Transfrom point from one space to another.
    """
    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError('Invalid input points p')

    # convert to homogeneous
    ps_homogeneous = np.hstack([ps, np.ones((len(ps), 1), dtype=np.float32)])
    ps_transformed = np.dot(t, ps_homogeneous.T).T

    return ps_transformed[:, :3]

def pointcloud(depth, fx, fy):
    #fy = fx = 0.5 / np.tan(fov * 0.5) # assume aspectRatio is one.
    height = depth.shape[0]
    width = depth.shape[1]

    mask = np.where(depth > 0)
    
    x = mask[1]
    y = mask[0]
    
    
    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = (y.astype(np.float32) - height * 0.5) / height
    
    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = depth[y, x]
    ones = np.ones(world_z.shape[0], dtype=np.float32)

    return np.vstack((world_x, world_y, world_z, ones)).T

def depth2boxfeatures(depth, fx, fy):
    point_cloud = pointcloud(depth, fx, fy)
    point_cloud = point_cloud[:, :3]
    #point_clouds = transform_point3s(camera2world, point_clouds)
    
    # max_min = np.array([np.max(point_clouds[:, 0]), np.max(point_clouds[:, 1]), np.max(point_clouds[:, 2]),
    #                     np.min(point_clouds[:, 0]), np.min(point_clouds[:, 1]), np.min(point_clouds[:, 2])])
    
    # boxfeatures = np.zeros((1, 6))
    # boxfeatures[:, -6:] = max_min
    boxfeatures = np.zeros((1, 10))
    obb = OBB.build_from_points(point_cloud)
    extents = obb.max - obb.min
    boxfeatures[:, :3] = extents
    boxfeatures[:, 3: 7] = matrix_to_quaternion(torch.tensor(obb.rotation.T)).numpy()
    boxfeatures[:, 7: 10] = obb.centroid.T
    #boxfeatures[:, -6:] = max_min
    return torch.tensor(boxfeatures).float()


def load_model(frames_input, save_dir='./save_model/'+"2021-12-07T00-00-00", model_name='checkpoint_60_0.000009.pth.tar'):
    encoder = Encoder(10, 512, frames_input, True)
    decoder = Decoder(10, 512 * 2)

    model_info = torch.load(os.path.join(save_dir, model_name))
    encoder.load_state_dict(model_info['enc_state_dict'])
    decoder.load_state_dict(model_info['dec_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder

def inference(encoder, decoder, inputs, frames_output):
    
    cam_pose_matrix = cam_view2pose([-0.7064125537872314, -0.4947664737701416, 0.5061494708061218, 0.0, 0.7078002691268921, -0.4937964379787445, 0.5051571130752563, 0.0, 0.0, 0.7151020765304565, 0.6990199089050293, 0.0, -0.0, -0.0, -3.386988639831543, 1.0])
    
    fx = 1.4441832304000854
    fy = 1.9255777597427368
    boxfeatures_list = [depth2boxfeatures(inputs[i], fx, fy, cam_pose_matrix) for i in range(len(inputs))]
    inputs = torch.unsqueeze(torch.cat(boxfeatures_list), dim=0)

    start_decode = torch.unsqueeze(inputs[:, -1, :], dim=1)
    input_lengths = (torch.ones(inputs.size()[0])*inputs.size()[1]).cpu()

    output, hidden_c = encoder(inputs, input_lengths)
    preds = decoder(start_decode, hidden_c, frames_output, output, None, is_training=False)
    return torch.squeeze(preds + torch.unsqueeze(inputs[:, 0, :], dim=1), dim=0)

encoder, decoder = load_model(3)
inputs = np.ones((3, 240, 320))
pred = inference(encoder, decoder, inputs, 1)


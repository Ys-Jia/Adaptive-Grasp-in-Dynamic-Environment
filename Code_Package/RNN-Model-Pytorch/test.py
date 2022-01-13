#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

import os
import json
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.mm import MovingMNIST, MovingObstacles, MovingObstaclesJSON
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
import random
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from seq2seq import Encoder, Decoder
from scipy.spatial.transform import Rotation as R
import trimesh
from pyobb.obb import OBB
import iou_3d

TIMESTAMP = "2021-12-10T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=1,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=3,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=1,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 199
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = './save_model/' + TIMESTAMP

with open('../dataset4.json', 'r') as f:
    depth_map = json.load(f)

folder_list = list(depth_map.keys())
indices = [i for i in range(len(folder_list)) if folder_list[i].split('_')[1] != "block" and folder_list[i].split('_')[1] != "teddy"]
vector_list = list(depth_map.values())
vector_list = [vector_list[idx] for idx in indices]
train_list = vector_list[:600]
test_list = vector_list[600:]
del depth_map

diff = np.zeros((1, 10))
count = 0
for i in range(len(vector_list)):
    vectors = list(vector_list[i].values())
    for j in range(len(vectors) - 1):
        diff += (np.asarray(vectors[j + 1]) - np.asarray(vectors[j]))
        count += 1
ratio = diff/count * 1000

validFolder = MovingObstaclesJSON(
                            test_list,
                            speed_scale=1,
                            is_train=False,
                            n_frames_input=args.frames_input,
                            n_frames_output=args.frames_output,
                            ratio=ratio
                            )

validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

def load_model(frames_input, save_dir='./save_model/'+"2021-12-10T00-00-00", model_name='checkpoint_19_0.000132.pth.tar'):
    encoder = Encoder(10, 512, frames_input, True)
    decoder = Decoder(10, 512 * 2)

    model_info = torch.load("/home/yuncong/COMS6998_datagenerator/ConvLSTM-PyTorch/save_model/2021-12-16T23-00-00/checkpoint_85_5.693271.pth.tar")
    encoder.load_state_dict(model_info['enc_state_dict'])
    decoder.load_state_dict(model_info['dec_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder

def inference(encoder, decoder, inputs, frames_output):
    
    # #cam_pose_matrix = cam_view2pose([-0.7064125537872314, -0.4947664737701416, 0.5061494708061218, 0.0, 0.7078002691268921, -0.4937964379787445, 0.5051571130752563, 0.0, 0.0, 0.7151020765304565, 0.6990199089050293, 0.0, -0.0, -0.0, -3.386988639831543, 1.0])
    
    # fx = 1.4441832304000854
    # fy = 1.9255777597427368
    # boxfeatures_list = [depth2boxfeatures(inputs[i], fx, fy, cam_pose_matrix) for i in range(len(inputs))]
    # inputs = torch.unsqueeze(torch.cat(boxfeatures_list), dim=0)
    base = torch.unsqueeze(inputs[:, 0, :], dim=1)
    inputs = inputs[:, 1:, :]
    start_decode = torch.unsqueeze(inputs[:, -1, :], dim=1)
    input_lengths = (torch.ones(inputs.size()[0])*inputs.size()[1]).cpu()

    output, hidden_c = encoder(inputs, input_lengths)
    preds = decoder(start_decode, hidden_c, frames_output, output, None, is_training=False)
    return preds*ratio + base

def vector2box(vector):
    vector = vector[0].float().numpy()
    extents = vector[:3]
    transformation = np.zeros((4, 4))
    transformation[3, 3] = 1
    transformation[:3, :3] = R.from_quat(vector[3:7].tolist()).as_matrix()
    transformation[:3, 3] = vector[-3:]
    #transformation = camera2world*transformation
    tri_mesh = trimesh.creation.box(extents, transformation)
    return tri_mesh

def vector2points(vector):
    x_corners = [vector[0], vector[0], vector[3], vector[3], vector[0], vector[0], vector[3], vector[3]]
    y_corners = [vector[1], vector[1], vector[1], vector[1], vector[4], vector[4], vector[4], vector[4]]
    z_corners = [vector[2], vector[5], vector[5], vector[2], vector[2], vector[5], vector[5], vector[2]]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def points2box(point_clouds):
    point_clouds = point_clouds[:, :3]
    obb = OBB.build_from_points(point_clouds)

    extents = obb.max - obb.min
    transformation = np.zeros((4, 4))
    transformation[3, 3] = 1
    transformation[:3, :3] = obb.rotation.T
    transformation[:3, 3] = obb.centroid.T
    #transformation = camera2world*transformation
    tri_mesh = trimesh.creation.box(extents, transformation)
    return tri_mesh

def test():
    '''
    main function to run the training
    '''
    encoder, decoder = load_model(args.frames_input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iou = 0
    count = 0
    with torch.no_grad():
        t = tqdm(validLoader, leave=False, total=len(validLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            if i == 3000:
                break
            inputs = inputVar.to(device)
            label = targetVar.to(device)

            pred = inference(encoder, decoder, inputs, args.frames_output)
            label = label*ratio + torch.unsqueeze(inputs[:, 0, :], dim=1)
            
            vector2box(pred[0]).export('pred_box.stl')
            vector2box(label[0]).export('label_box.stl')
            break
            #iou_3d, _ = box3d_iou(vector2points(pred[0][3]), vector2points(label[0][3]))
            #iou += iou_3d
            #count += 1
            
    #print(iou/count)
    # for i in range(len(pred)):
    #     input_tmp = np.asarray(torch.squeeze(inputs[i][-1]).cpu())
    #     pred_tmp = np.asarray(torch.squeeze(pred[i]).cpu())
    #     label_tmp = np.asarray(torch.squeeze(label[i]).cpu())
    #     diff_tmp = np.abs(pred_tmp - label_tmp)
    #     baseline_tmp = np.abs(input_tmp - label_tmp)
    #     write_depth(pred_tmp*(label_tmp > 0), f'visualization/pred_{i}.png')
    #     write_depth(label_tmp, f'visualization/label_{i}.png')
    #     write_depth(diff_tmp, f'visualization/diff_{i}.png')
    #     write_depth(baseline_tmp, f'visualization/baseline_{i}.png')
    #     # plt.imsave(f'visualization/pred{i}.png', pred_tmp)
    #     # plt.imsave(f'visualization/label{i}.png', label_tmp)
    #     # plt.imsave(f'visualization/diff{i}.png', diff_tmp)
    #     # plt.imsave(f'visualization/base{i}.png', baseline_tmp)

import numpy as np
from scipy.spatial import ConvexHull
from numpy import *

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    #iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, 0

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d



if __name__ == "__main__":
    test()
import numpy as np
import math

from numpy.lib.twodim_base import tri
import cv2
import open3d as o3d
import trimesh
import pymeshfix
import time
import torch

from pytorch3d.transforms import matrix_to_quaternion

from pyobb.obb import OBB
from scipy.spatial.transform import Rotation as R
from inference_obstacle import *

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

def points2mesh(point_clouds):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds[:, :3])
    pcd.estimate_normals()

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               pcd,
               o3d.utility.DoubleVector([radius, radius * 2]))

    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                              vertex_normals=np.asarray(mesh.vertex_normals))

    trimesh.convex.is_convex(tri_mesh)
    return tri_mesh

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

def read_mask(file_path):
    """
    In:
        file_path: Path to mask png image saved as 8-bit integer.
    Out:
        Segmentation mask as np array [height, width].
    Purpose:
        Read in a mask image.
    """
    return cv2.imread(file_path, -1)

def depth2mesh(depth, fx, fy, camera2world, mesh=True):
    point_cloud = pointcloud(depth, fx, fy)
    point_cloud = point_cloud[:, :3]
    
    world_point_cloud = transform_point3s(camera2world, point_cloud)

    if mesh:
        tri_mesh = points2mesh(world_point_cloud)
    else:
        tri_mesh = points2box(world_point_cloud)
    
    return tri_mesh
    
def depth2boxfeatures(depth, fx, fy, camera2world):
    point_cloud = pointcloud(depth, fx, fy)
    point_clouds = point_cloud[:, :3]
    point_clouds = transform_point3s(camera2world, point_clouds)
    
    max_min = np.array([np.max(point_clouds[:, 0]), np.max(point_clouds[:, 1]), np.max(point_clouds[:, 2]),
                        np.min(point_clouds[:, 0]), np.min(point_clouds[:, 1]), np.min(point_clouds[:, 2])])
    
    obb = OBB.build_from_points(point_clouds)
    
    boxfeatures = np.zeros((1, 16))
    extents = obb.max - obb.min
    boxfeatures[:, :3] = extents
    boxfeatures[:, 3: 7] = R.from_matrix(obb.rotation.T.tolist()).as_quat()
    boxfeatures[:, 7: 10] = obb.centroid
    boxfeatures[:, -6:] = max_min
    return torch.tensor(boxfeatures)
    

def read_depth(file_path):
    """
    In:
        file_path: Path to depth png image saved as 16-bit z depth in mm.
    Out:
        depth_image: np array [height, width].
    Purpose:
        Read in a depth image.
    """
    # depth is saved as 16-bit uint in millimenters
    depth_image = cv2.imread(file_path, -1).astype(float)

    # millimeters to meters
    depth_image /= 1000.

    return depth_image

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

def get_intrinsic_matrix(fx, fy, h, w):
    intrinsic_matrix = np.eye(3)  # TODO: compute camera intrinsic_matrix from camera parameters
    intrinsic_matrix[0, 0] = fx
    intrinsic_matrix[1, 1] = fy
    intrinsic_matrix[0, 2] = w/2
    intrinsic_matrix[1, 2] = h/2

fx = 1.4441832304000854
fy = 1.9255777597427368

cam_pose_matrix = cam_view2pose([-0.7064125537872314, -0.4947664737701416, 0.5061494708061218, 0.0, 0.7078002691268921, -0.4937964379787445, 0.5051571130752563, 0.0, 0.0, 0.7151020765304565, 0.6990199089050293, 0.0, -0.0, -0.0, -3.386988639831543, 1.0])
intrinsic_matrix = get_intrinsic_matrix(fx, fy, 240, 320)
label = read_depth("/home/yuncong/COMS6998_datagenerator/Project/dataset/22_duck_circular_0.75/depth_76.png")
mask = read_mask("/home/yuncong/COMS6998_datagenerator/Project/dataset/22_duck_circular_0.75/mask_76.png")

label = (mask > 0)*label



start_time = time.time()
# #box = depth2mesh(label, fx, fy, cam_pose_matrix, mesh=False)
# #label = depth2mesh(label, fx, fy, cam_pose_matrix, mesh=True)
# print(time.time() - start_time)
# label.export('label.stl')
# box.export('box.stl')
print(depth2boxfeatures(label, fx, fy, cam_pose_matrix))

print(cam_pose_matrix)

def make_observation(self):
    far, near = 10.0, 0.01
    _, depth_obs, mask_obs = self.camera.getCameraImage()
    depth_obs = (far * near / (far - (far - near) * depth_obs)) * (mask_obs == self.obstacle.id)
    self.depth_maps.append(np.expand_dims(depth_obs, axis=0))
    self.object_pose.append(np.expand_dims(self.target.get_pose()[0], axis=0))
# # # tin = pymeshfix.PyTMesh()

# # # tin.load_array(np.asarray(mesh.vertices), np.asarray(mesh.triangles)) # or read arrays from memory
# # # #tin.fill_small_boundaries()

# # # #tin.clean(max_iters=5, inner_loops=3)
# # # tin.save_file('out.ply')

# # # meshfix = pymeshfix.MeshFix(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
# # # meshfix.repair()
# # # meshfix.write('out.ply')
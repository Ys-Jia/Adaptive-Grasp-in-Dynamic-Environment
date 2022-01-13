import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
from torchvision import transforms
import cv2
import numpy as np
import trimesh
from pyobb.obb import OBB
from pytorch3d.transforms import matrix_to_quaternion

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
    point_clouds = point_cloud[:, :3]
    obb = OBB.build_from_points(point_clouds)
    
    boxfeatures = np.zeros((1, 15))
    extents = obb.max - obb.min
    boxfeatures[:, :3] = extents
    boxfeatures[:, 3: 12] = obb.rotation.reshape(9,)
    boxfeatures[:, 12: 15] = obb.centroid
    return torch.tensor(boxfeatures)

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist

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


def load_fixed_set(root, is_train):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset

class MovingObstaclesJSON(data.Dataset):
    def __init__(self, vector_list, speed_scale, is_train, n_frames_input, n_frames_output, ratio):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingObstaclesJSON, self).__init__()
        self.speed_scale = speed_scale
        
        self.folders = vector_list
        self.ratio = ratio

        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        depth_imgs = list(self.folders[idx % len(self.folders)].values())
        start_index = random.randint(0, len(depth_imgs) - 1 - self.speed_scale*length)
        indices = [start_index + i*self.speed_scale for i in range(length)]
        depth_imgs = [torch.tensor(depth_imgs[index]) for index in indices]
        #mask_imgs = [read_mask(img) for img in mask_imgs]
        #depth_imgs = [read_depth(img) for img in depth_imgs]
        #if self.transform is not None:
            #images = self.transform(images)
        # depth_new = [torch.zeros(1, 9) for i in range(len(depth_imgs))]
        # for i in range(len(depth_imgs)):
        #     depth_new[i][:, :] = depth_imgs[i][:, :9]
        # depth_new = [torch.zeros(1, 6) for i in range(len(depth_imgs))]
        # for i in range(len(depth_imgs)):
        #     depth_new[i][:, :] = depth_imgs[i][:, -6:]
        
        #depth_new = [depth_imgs[0]] + [depth_imgs[i] - depth_imgs[0] for i in range(len(depth_imgs))]
        depth_new = [depth_imgs[i] - depth_imgs[0] for i in range(len(depth_imgs))]
        depth_new = [depth/self.ratio for depth in depth_new]
        #depth_new = [depth_imgs[0]] + [depth/self.ratio for depth in depth_new]
        depth_imgs = torch.cat(depth_new)
        images = depth_imgs
        
        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        frozen = input[-1]

        output = output.float()
        input = input.float()

        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return len(self.folders)*10

class MovingObstacles(data.Dataset):
    def __init__(self, root, folders, speed_scale, is_train, n_frames_input, n_frames_output,
                 transform=transforms.ToTensor()):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingObstacles, self).__init__()

        self.data_dir = root
        self.folders = folders
        self.speed_scale = speed_scale

        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        imgs = os.listdir(os.path.join(self.data_dir, self.folders[idx]))
        depth_imgs = [img for img in imgs if img.split('_')[0] == 'depth']
        mask_imgs = [img for img in imgs if img.split('_')[0] == 'mask']
        depth_imgs.sort(key = lambda img: int(img.split('_')[1].split('.')[0]))
        mask_imgs.sort(key = lambda img: int(img.split('_')[1].split('.')[0]))
        start_index = random.randint(0, len(depth_imgs) - 1 - self.speed_scale*length)
        indices = [start_index + i*self.speed_scale for i in range(length)]
        mask_imgs = [read_mask(os.path.join(self.data_dir, self.folders[idx], mask_imgs[index])) for index in indices]
        depth_imgs = [read_depth(os.path.join(self.data_dir, self.folders[idx], depth_imgs[index])) for index in indices]
        #mask_imgs = [read_mask(img) for img in mask_imgs]
        #depth_imgs = [read_depth(img) for img in depth_imgs]
        #if self.transform is not None:
            #images = self.transform(images)
        mask_imgs = [x > 0 for x in mask_imgs]
        depth_imgs = [mask_imgs[i]*depth_imgs[i] for i in range(len(mask_imgs))]
        
        fx = 1.4441832304000854
        fy = 1.9255777597427368
        depth_imgs = [depth2boxfeatures(depth_imgs[i], fx, fy) - depth2boxfeatures(depth_imgs[0], fx, fy) for i in range(len(depth_imgs))]
        depth_imgs = torch.cat(depth_imgs)
        images = depth_imgs
        
        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        frozen = input[-1]

        output = output.float()
        input = input.float()

        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return len(self.folders)

class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
                 transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root, False)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        # if self.transform is not None:
        #     images = self.transform(images)

        r = 1
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        frozen = input[-1]
        # add a wall to input data
        # pad = np.zeros_like(input[:, 0])
        # pad[:, 0] = 1
        # pad[:, pad.shape[1] - 1] = 1
        # pad[:, :, 0] = 1
        # pad[:, :, pad.shape[2] - 1] = 1
        #
        # input = np.concatenate((input, np.expand_dims(pad, 1)), 1)

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        # print()
        # print(input.size())
        # print(output.size())

        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return self.length

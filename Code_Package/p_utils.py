import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, sqrt, atan, radians, degrees
from collections import namedtuple
from rrt.rrt import RRT
from rrt.search_space import SearchSpace

file_path = "D:\Learning\\Newlearning\\COMS6998" # need to change!!

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])
BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])

class KUKA(object):
    def __init__(self, basepose, baseorient, EndEffector=6):
        self.Kukapath = "kuka_iiwa/model.urdf" # need to change to your path
        self.startpose = basepose
        self.startorient = p.getQuaternionFromEuler(baseorient)
        self.KukaEndEffectorIndex = EndEffector

    def initial(self):
        self.load_KUKA()
        self.get_rot_joints()
        self.reset_pos()

    def load_KUKA(self):
        self.Kuka = p.loadURDF(self.Kukapath, self.startpose, self.startorient)

    def get_rot_joints(self, putinGUI=0, set_range=10, initial=0):
        self.number_joints = p.getNumJoints(self.Kuka)
        self.jointIds = []
        self.paramIds = []
        for j in range(self.number_joints):
            info = p.getJointInfo(self.Kuka, j)
            jointName = info[1]; jointType = info[2]
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                self.jointIds.append(j)
                if putinGUI:
                    self.paramIds.append(p.addUserDebugParameter(jointName.decode('utf-8'), -set_range, set_range, initial))

    def set_joints(self, pos, orient=None, speed=0):
        if orient is not None: orient = p.getQuaternionFromEuler(orient)
        if type(pos) == list:  # list type means position, tuble type means joints' orientations to control
            jointPoses = p.calculateInverseKinematics(self.Kuka, self.KukaEndEffectorIndex, pos, orient)
        else:
            jointPoses = pos

        assert len(self.jointIds) == len(jointPoses), 'Number of Joints must equal to Number of positions'
        old_time = time.time()
        if speed:
            p.setJointMotorControlArray(self.Kuka, self.jointIds, p.POSITION_CONTROL, jointPoses,
                                        positionGains=speed * np.ones_like(self.jointIds))
        else:
            p.setJointMotorControlArray(self.Kuka, self.jointIds, p.POSITION_CONTROL, jointPoses)
        return jointPoses

    def currentstate(self): # return each joint's position
        return np.array([p.getJointState(self.Kuka, i)[0] for i in self.jointIds])

    def endpose(self):
        return (p.getLinkState(self.Kuka, self.KukaEndEffectorIndex)[0], p.getEulerFromQuaternion(p.getLinkState(self.Kuka, self.KukaEndEffectorIndex)[1]))

    def reset_pos(self):
        p.resetBasePositionAndOrientation(self.Kuka, self.startpose, self.startorient)

class Conveyor:
    def __init__(self, initial_pose, urdf_path=file_path+'\my_file\\urdf\conveyor.urdf'):
        self.initial_pose = initial_pose
        self.urdf_path = urdf_path
        self.id = p.loadURDF(self.urdf_path, initial_pose[0], initial_pose[1])

        self.cid = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                      childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0], childFramePosition=initial_pose[0],
                                      childFrameOrientation=initial_pose[1])

        # motion related
        self.start_pose = None
        self.target_pose = None
        self.discretized_trajectory = None
        self.wp_target_index = 0
        self.distance = None
        self.theta = None
        self.length = None
        self.direction = None
        self.speed = None
        self.z_start = None
        self.z_end = None
        self.circular_angles = None

    def reset(self):
        set_pose(self.id, self.initial_pose)

    def set_pose(self, pose):
        set_pose(self.id, pose)
        self.control_pose(pose)

    def get_pose(self):
        return get_pose(self.id)

    def control_pose(self, pose): # real move control not reset
        p.changeConstraint(self.cid, jointChildPivot=pose[0], jointChildFrameOrientation=pose[1])

    def step(self):
        if self.discretized_trajectory is None or self.wp_target_index == len(self.discretized_trajectory):
            pass
        else:
            self.control_pose(self.discretized_trajectory[self.wp_target_index])
            self.wp_target_index += 1

    def initialize_linear_motion(self, dist, theta, length, direction, speed, z_end, variable_speed=False, is_random=False):
        """
        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 360), angle of start point to robot center
        :param length: the length of the motion
        :param direction: the direction of the motion
            1: from smaller theta to larger theta
            -1: from larger theta to smaller theta
        :param z_start: the height of the conveyor at the start
        :param z_end: the height of the conveyor at the end
        :param speed: the speed of the conveyor
        :param variable_speed: determines if the speed of the conveyor is variable or constant
        """
        self.distance = float(dist)
        self.theta = float(theta)
        self.length = float(length)
        self.direction = float(direction)
        self.speed = float(speed)
        self.z_start = float(self.initial_pose[0][2]) # same as initial pose
        self.z_end = float(z_end)
        # use the orientation of the initial pose
        orientation = self.initial_pose[1]
        # compute start xy and end xy
        new_dist = sqrt(dist ** 2 + (length / 2.0) ** 2)  # start point to robot center
        delta_theta = atan((length / 2.0) / dist) # angle of start point to robot center

        theta_large = self.theta + delta_theta
        theta_small = self.theta - delta_theta

        if direction == -1:
            start_xy = [new_dist * cos(theta_large), new_dist * sin(theta_large)]
            target_xy = [new_dist * cos(theta_small), new_dist * sin(theta_small)]
        elif direction == 1:
            target_xy = [new_dist * cos(theta_large), new_dist * sin(theta_large)]
            start_xy = [new_dist * cos(theta_small), new_dist * sin(theta_small)]
        else:
            raise ValueError('direction must be in {-1, 1}')
        start_position = start_xy + [self.z_start]
        target_position = target_xy + [self.z_end]

        self.start_pose = [start_position, orientation]
        self.target_pose = [target_position, orientation]

        if variable_speed:
            n_segments = 10  # num speed switches
            speed_multipliers = np.linspace(0.6, 1.0, n_segments)[::-1]
            speeds = speed_multipliers * self.speed
            segments = np.linspace(start_position, target_position, n_segments+1)
            position_trajectory = []
            for i in range(n_segments):
                # speed = np.random.choice(speeds)
                speed = speeds[i]
                dist = np.linalg.norm(segments[i] - segments[i+1])
                num_steps = int(dist / speed * 240)
                wps = np.linspace(segments[i], segments[i+1], num_steps)
                position_trajectory.extend(wps)
        else:
            num_steps = int(self.length / self.speed * 240)
            position_trajectory = np.linspace(start_position, target_position, num_steps)

        if is_random:
            self.discretized_trajectory = [[list(pos+np.random.uniform(-0.05, 0.05, 3)), orientation] for pos in position_trajectory]
        else:
            self.discretized_trajectory = [[list(pos), orientation] for pos in position_trajectory]
        self.wp_target_index = 1
        self.set_pose(self.discretized_trajectory[0]) # move to start point

    def initialize_sinusoid_motion(self, dist, theta, length, direction, speed, amp_div=8, period_div=3):
        """
        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 360)
        :param length: the length of the motion
        :param direction: the direction of the motion
            1: from smaller theta to larger theta
            -1: from larger theta to smaller theta
        :param speed: the speed of the conveyor
        """
        self.distance = float(dist)
        self.theta = float(theta)
        self.length = float(length)
        self.direction = float(direction)
        self.speed = float(speed)
        # uses the z value and orientation of the current pose
        z = self.get_pose()[0][-1]
        orientation = self.get_pose()[1]

        num_steps = int(self.length / self.speed * 240)

        start_position = np.array([0, -length / 2.0, 1]) * direction
        target_position = np.array([0, length / 2.0, 1]) * direction
        position_trajectory = np.linspace(start_position, target_position, num_steps)
        # Amplitude: length/4., period: self.length/3 i.e. 3 sinusoids within the length of the trajectory
        position_trajectory[:, 0] = (self.length * 1.0 / amp_div) * np.sin(2 * np.pi * position_trajectory[:, 1] /
                                                                           (self.length * 1.0 / period_div))
        T_1 = np.array([[np.cos(radians(self.theta)), -np.sin(radians(self.theta)), 0],
                        [np.sin(radians(self.theta)), np.cos(radians(self.theta)), 0],
                        [0, 0, 1]])
        T_2 = np.array([[1, 0, self.distance], [0, 1, 0], [0, 0, 1]])
        position_trajectory = np.dot(T_1, np.dot(T_2, position_trajectory.T)).T
        position_trajectory[:, -1] = z
        self.start_pose = [position_trajectory[0], orientation]
        self.target_pose = [position_trajectory[-1], orientation]

        self.discretized_trajectory = [[list(pos), orientation] for pos in position_trajectory]
        self.wp_target_index = 1
        self.set_pose(self.discretized_trajectory[0])  # move to start point

    def initialize_circular_motion(self, dist, theta, length, direction, speed, is_random=False):
        """
        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 360)
        :param length: the length of the motion
        :param direction: the direction of the motion
            1: counter clockwise
            -1: clockwise
        :param speed: the speed of the conveyor
        """
        self.distance = float(dist)
        self.theta = float(theta)
        self.length = float(length)
        self.direction = float(direction)
        self.speed = float(speed)
        # uses the z value and orientation of the current pose
        z = self.get_pose()[0][-1]
        orientation = self.get_pose()[1]

        # calculate waypoints
        num_points = int(self.length / self.speed * 240)
        delta_angle = self.length / self.distance
        angles = np.linspace(radians(theta), radians(theta)+delta_angle, num_points)
        if direction == -1:
            angles = angles[::-1]
        self.circular_angles = angles
        if is_random:
            self.discretized_trajectory = [[[cos(ang) * self.distance, sin(ang) * self.distance, z+np.random.normal()/2], orientation] for ang in angles]
        else:
            self.discretized_trajectory = [[[cos(ang) * self.distance, sin(ang) * self.distance, z], orientation] for ang in angles]
        self.wp_target_index = 1

        self.start_pose = self.discretized_trajectory[0]
        self.target_pose = self.discretized_trajectory[-1]
        self.set_pose(self.discretized_trajectory[0])  # move to start point

    def initialize_circlearound_motion(self, radiaus, speed, length, direction, start_pose=None):
        """
        Initialize a motion using the start pose as initial pose, in the direction of the angle.

        :param angle: the angle of the motion direction in the conveyor frame, in degrees
        :param speed: the speed of the motion
        """
        self.radiaus = float(radiaus)
        self.length = float(length)
        self.speed = float(speed)
        self.direction = float(direction)
        start_pose = self.initial_pose if start_pose is None else start_pose

        num_points = int(self.length / self.speed * 240)
        delta_angle = self.length / self.radiaus
        angles = np.linspace(radians(0), delta_angle, num_points)

        if direction == -1:
            angles = angles[::-1]
        self.circular_angles = angles
        self.discretized_trajectory = [[[start_pose[0][0], cos(ang) * self.radiaus + start_pose[0][1], sin(ang) * self.radiaus + start_pose[0][2]], start_pose[1]] for ang in angles]
        self.wp_target_index = 1

        self.start_pose = self.discretized_trajectory[0]
        self.target_pose = self.discretized_trajectory[-1]
        self.set_pose(self.discretized_trajectory[0])  # move to start point

    def check_done(self):
        return self.wp_target_index >= len(self.discretized_trajectory)

    def clear_motion(self):
        self.start_pose = None
        self.target_pose = None
        self.discretized_trajectory = None
        self.wp_target_index = 0
        self.distance = None
        self.theta = None
        self.length = None
        self.direction = None
        self.circular_angles = None

    def predict(self, duration):
        # predict the ground truth future pose of the conveyor
        num_predicted_steps = int(duration * 240)
        predicted_step_index = self.wp_target_index - 1 + num_predicted_steps
        if predicted_step_index < len(self.discretized_trajectory):
            return self.discretized_trajectory[predicted_step_index]
        else:
            return self.discretized_trajectory[-1]

class Targetbody(Conveyor):
    def __init__(self, path, initial_pose=[[0.5, 0.5, 0], [0, 0, 0, 1]], scale=1):
        self.path = path
        self.initial_pose = initial_pose
        self.id = p.loadURDF(self.path, self.initial_pose[0], self.initial_pose[1], globalScaling=scale)
        self.oldposition = initial_pose[0]

    def puton_conveyor(self, conveyor):
        assert conveyor.discretized_trajectory is not None, 'must plan conveyor trajectory first!'
        target_startpose = conveyor.discretized_trajectory[0];
        target_startpose[0][2] += 0.03  # cube setting on the conveyor, must do!
        set_pose(self.id, target_startpose)

    def add_line(self, color=[0,0,0]):
        if self.oldposition == self.initial_pose[0]: self.oldposition = self.get_pose()[0]; return # initialize first position
        p.addUserDebugLine(self.oldposition, self.get_pose()[0], lineColorRGB=color)
        self.oldposition = self.get_pose()[0]

    def reset(self):
        set_pose(self.id, self.initial_pose)
        self.oldposition = self.initial_pose[0]

class Obstaclebody(Targetbody):
    def __init__(self, path, initial_pose=[[0.5, 0.5, 0.5], [0, 0, 0, 1]], scale=1):
        self.path = path
        self.initial_pose = initial_pose
        self.id = p.loadURDF(self.path, self.initial_pose[0], self.initial_pose[1], globalScaling=scale)
        self.cid = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                      childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0], childFramePosition=initial_pose[0],
                                      childFrameOrientation=initial_pose[1])
        self.oldposition = initial_pose[0]
        self.theta = 0

    def add_line(self, color=[0,0,0]):
        if self.oldposition == self.initial_pose[0]: self.oldposition = self.get_pose()[0]; return # initialize first position
        if self.wp_target_index == 2: self.oldposition = self.discretized_trajectory[0][0]
        p.addUserDebugLine(self.oldposition, self.get_pose()[0], lineColorRGB=color)
        self.oldposition = self.get_pose()[0]

class Camera(object):
    def __init__(self):
        self.viewMatrix = (-0.6996635794639587, 0.5052081942558289, -0.5052083134651184, 0.0, -0.7144722938537598, -0.49473685026168823, 0.4947369694709778, 0.0, 0.0, 0.7071068286895752, 0.7071067690849304, 0.0, 0.17047584056854248, -0.08954763412475586, -1.7710728645324707, 1.0)
        self.projection = 0
        self.buttonid = p.addUserDebugParameter('Camera Pose Save', 1, 0, 0)

    def set_camera_pose(self):
        if p.readUserDebugParameter(self.buttonid) == 1:
            self.save_Matrix()

    def getCameraImage(self, width=512, height=512):
        self.resolution = (width, height)
        _, _, RGB, DEP, SEG = p.getCameraImage(width, height, self.viewMatrix, self.projection)
        self.package = (RGB, DEP, SEG)
        return self.package

    def plot(self, type=0):
        plt.imshow(self.package[type])
        plt.show()

    def save_Matrix(self):
        self.viewMatrix = p.getDebugVisualizerCamera()[2]
        self.projection = p.getDebugVisualizerCamera()[3]
        matrix = [self.viewMatrix, self.projection]
        np.savetxt('Matrix.txt', matrix)
        print('Save view matrix sucessfully')

    def load_Matrix(self, loadpath=None):
        if loadpath is None:
            loadpath = 'Matrix.txt'
        self.viewMatrix, self.projection = np.loadtxt(loadpath)
        print('Load view matrix sucessfully')

class UR5_whole():
    JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                         'qIndex', 'uIndex', 'flags',
                                         'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                         'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                         'parentFramePos', 'parentFrameOrn', 'parentIndex'])

    JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                           'jointReactionForces', 'appliedJointMotorTorque'])

    LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                         'localInertialFramePosition', 'localInertialFrameOrientation',
                                         'worldLinkFramePosition', 'worldLinkFrameOrientation'])

    # movable joints for each moveit group
    GROUPS = {
        'arm': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
        'gripper': ['finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint', 'right_outer_knuckle_joint',
                    'right_inner_knuckle_joint', 'right_inner_finger_joint']
    }
    HOME = [0, -0.8227210029571718, -0.130, -0.660, 0, 1.62]
    # HOME = [0, -1.15, 0.9, -0.660, 0, 0.0]
    HOME_POS = (0.6109604239463806, 0.31644999980926514, 0.7443326711654663)
    HOME_QUT =  (-0.7045285105705261, -2.0915123855047568e-08, 0.7096756100654602, 2.0915123855047568e-08)
    OPEN_POSITION = [0] * 6
    CLOSED_POSITION = 0.72 * np.array([1, 1, -1, 1, 1, -1]) # how to get?
    TRANS = 0.11968471932244991 # three point one line transmision
    TRANS_EE = 0.07216878394698116

    JOINT_INDICES_DICT = {}
    EE_LINK_NAME = 'ee_link'
    WRIST_LINK = 'wrist_3_link'
    TIP_LINK = "robotiq_arg2f_base_link"
    BASE_LINK = "base_link"
    ARM = "manipulator"
    GRIPPER = "gripper"
    ARM_JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    GRIPPER_JOINT_NAMES = ['finger_joint']

    # this is read from moveit_configs joint_limits.yaml
    MOVEIT_ARM_MAX_VELOCITY = [3.15, 3.15, 3.15, 3.15, 3.15, 3.15]
    def __init__(self):
        self.URpath = file_path + "\dynamic_grasping_assets\\ur5\\ur5_robotiq.urdf"  # need to change to your path
        self.id = p.loadURDF(self.URpath, [0, 0, 0.02], p.getQuaternionFromEuler([0, 0, 0]), flags=p.URDF_USE_SELF_COLLISION)
        self.num_joints = p.getNumJoints(self.id)
        joint_infos = [p.getJointInfo(self.id, joint_index) for joint_index in range(p.getNumJoints(self.id))]
        self.JOINT_INDICES_DICT = {entry[1].decode(): entry[0] for entry in joint_infos}
        self.GROUP_INDEX = {key: [self.JOINT_INDICES_DICT[joint_name] for joint_name in self.GROUPS[key]] for key in
                            self.GROUPS}
        self.EEF_LINK_INDEX = link_from_name(self.id, self.EE_LINK_NAME)
        self.WRIST_LINK_INDEX = link_from_name(self.id, self.WRIST_LINK)
        self.arm_difference_fn = get_difference_fn(self.id, self.GROUP_INDEX['arm'])
        self.arm_max_joint_velocities = [get_max_velocity(self.id, j_id) for j_id in self.GROUP_INDEX['arm']]
        self.reset()
        self.oldpose = self.get_link_state(self.EEF_LINK_INDEX).linkWorldPosition

    def reset(self):
        self.set_arm_joints(self.HOME)
        self.set_gripper_joints(self.OPEN_POSITION)
        self.arm_discretized_plan = None
        self.gripper_discretized_plan = None
        self.arm_wp_target_index = 0
        self.gripper_wp_target_index = 0

    def set_arm_joints(self, joint_values):
        set_joint_positions(self.id, self.GROUP_INDEX['arm'], joint_values)
        control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def control_arm_joints(self, joint_values):
        control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def set_gripper_joints(self, joint_values):
        set_joint_positions(self.id, self.GROUP_INDEX['gripper'], joint_values)
        control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values)

    def control_gripper_joints(self, joint_values):
        control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values)

    def close_gripper(self, duration=0.1):
        self.plan_gripper_joint_values(self.CLOSED_POSITION, duration=duration)

    def open_gripper(self, duration=0.1):
        self.plan_gripper_joint_values(self.OPEN_POSITION, duration=duration)

    def plan_arm_joint_values(self, goal_joint_values, start_joint_values=None, duration=None):
        """ Linear interpolation between joint_values """
        start_joint_values = self.get_arm_joint_values() if start_joint_values is None else start_joint_values

        diffs = self.arm_difference_fn(goal_joint_values, start_joint_values)

        steps = np.abs(np.divide(diffs, self.MOVEIT_ARM_MAX_VELOCITY)) * 240
        # num_steps = int(max(steps)) if int(max(steps)) != 0 else 1
        num_steps = 2 # 0.2s
        if duration is not None:
            num_steps = int(duration * 240)
            # num_steps = max(int(duration * 240), steps)     # this should ensure that it satisfies the max velocity of the end-effector

        goal_joint_values = np.array(start_joint_values) + np.array(diffs)
        self.arm_discretized_plan = np.linspace(start_joint_values, goal_joint_values, num_steps)
        self.arm_wp_target_index = 1

    def plan_gripper_joint_values(self, goal_joint_values, start_joint_values=None, duration=None): # 1s grasp?
        start_joint_values = self.get_gripper_joint_values() if start_joint_values is None else start_joint_values
        num_steps = 240 if duration is None else int(duration*240)
        self.gripper_discretized_plan = np.linspace(start_joint_values, goal_joint_values, num_steps)
        self.gripper_wp_target_index = 1

    def gripper2endeffector(self, gripperpose, gripperorient=None): # transform pose/orientation to effector 7 for inverse solution #????????????
        if gripperorient is None: gripperorient = self.get_link_state(link_from_name(self.id, self.TIP_LINK)).linkWorldOrientation
        offset = [0.125, 0, 0]  # same as combine gripper's initial setting
        orientation = p.getQuaternionFromEuler([0, -np.pi/2, np.pi/2])
        invert_offset, invert_orient = p.invertTransform(offset, orientation)
        real_pose, real_orient = p.multiplyTransforms(gripperpose, gripperorient, invert_offset, invert_orient) # real means endeffector 7 pose, orientation
        return real_pose, real_orient

    def endeffector2gripper(self):
        EEF = self.get_link_state(self.EEF_LINK_INDEX)
        endpose, endorient = EEF.linkWorldPosition, EEF.linkWorldOrientation
        offset = [0.125, 0, 0]; orientation = p.getQuaternionFromEuler([0, -np.pi/2, np.pi/2])
        gripperpose, gripperorient = p.multiplyTransforms(endpose, endorient, offset, orientation)
        return gripperpose, gripperorient

    def ik_pybullet(self, pos, orient):
        pos, orient = self.gripper2endeffector(pos, orient)
        arm_joint_values = self.get_arm_joint_values()
        gripper_joint_values = self.get_gripper_joint_values()
        return p.calculateInverseKinematics(self.id, self.EEF_LINK_INDEX, pos, orient)[:len(self.GROUPS['arm'])]

    def ik_wrist(self, targetpos, steppos, orient=None):
        vector = np.array(steppos) - np.array(targetpos)
        D = distance(targetpos, steppos)
        self.wrist_pos = np.array(steppos) - (self.TRANS/D) * vector
        self.ee_pos = np.array(steppos) - (self.TRANS_EE/D) * vector
        return p.calculateInverseKinematics(self.id, link_from_name(self.id, self.WRIST_LINK), self.wrist_pos, orient)[:len(self.GROUPS['arm'])-1]

    def ik_fastpy(self, pos, orient, all_id, filter=True):
        pos, orient = self.gripper2endeffector(pos, orient)
        vector = list(pos) + list(orient)
        solutions = np.asarray(self.kinematics.inverse(vector)).reshape(-1, self.arm_joints)

        # gripper_pose = np.zeros((3, 4))
        # gripper_pose[:, :3] = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
        # gripper_pose[:, 3] = pos
        # solutions = np.asarray(self.kinematics.inverse(gripper_pose.reshape(-1).tolist())).reshape(-1, self.arm_joints)
        if filter:
            filter_solutions = []
            for solution in solutions:
                Flag, error_dis = self.filter_ik(solution, all_id)
                if not Flag: filter_solutions.append(list(solution)+error_dis)
            filter_solutions = np.array(filter_solutions)
            if len(filter_solutions) > 0: filter_solutions = filter_solutions[filter_solutions[:, -1].argsort()][:, :-1] # remove last column
        else: filter_solutions = solutions
        return filter_solutions

    def filter_ik(self, joint_values, all_id):
        old_values = self.get_arm_joint_values()
        self.set_arm_joints(joint_values)
        Flag = self.collision_check(all_id)
        self.set_arm_joints(old_values)
        return Flag

    def add_line(self, color=[1, 0, 0]):
        new_pose = self.endeffector2gripper()[0]
        p.addUserDebugLine(self.oldpose, new_pose, lineColorRGB=color, lineWidth=0.2)
        self.oldpose = new_pose

    def collision_check(self, worldids):
        for id in worldids:
            temp = closest(self.id, id)
            if temp is None: continue # None means disance is more than 1
            elif temp < 0.003: return True
        return False

    def get_joint_state(self, joint_index):
        return self.JointState(*p.getJointState(self.id, joint_index))

    def get_link_state(self, link_index):
        return self.LinkState(*p.getLinkState(self.id, link_index))

    def get_arm_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['arm']]

    def get_gripper_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['gripper']]

    def step(self):
        """ step the robot for 1/240 second """
        # calculate the latest conf and control array
        if self.arm_discretized_plan is None or self.arm_wp_target_index == len(self.arm_discretized_plan): pass
        else:
            self.control_arm_joints(self.arm_discretized_plan[self.arm_wp_target_index])
            self.arm_wp_target_index += 1

        if self.gripper_discretized_plan is None or self.gripper_wp_target_index == len(self.gripper_discretized_plan): pass
        else:
            self.control_gripper_joints(self.gripper_discretized_plan[self.gripper_wp_target_index])
            self.gripper_wp_target_index += 1

    def arm_finished(self): return self.arm_wp_target_index == len(self.arm_discretized_plan)
    def gripper_finished(self): return self.gripper_wp_target_index == len(self.gripper_discretized_plan)

    def execute_gripper(self): # direct complete gripper
        while not self.gripper_finished():
            self.control_gripper_joints(self.gripper_discretized_plan[self.gripper_wp_target_index])
            self.gripper_wp_target_index += 1
            steps(1)

    def execute_grasp(self):
        while not self.arm_finished():
            self.control_arm_joints(self.arm_discretized_plan[self.arm_wp_target_index])
            self.arm_wp_target_index += 1
            steps(2)

    def get_closest_obs(self, obstacle_box):
        result_box = []
        for obs in obstacle_box:  # not include ground but needs to include conveyor in test
            pos_obstacle = obs[0]
            orient_obstacle = obs[1]
            extension = obs[2]
            id1, id2 = create_cube(pos_obstacle, orient_obstacle, extension)
            close_dis = closest(self.id, id2, 12)
            result_box.append(pos_obstacle + [close_dis])
            del_cube(id1, id2)
        result_box = np.array(result_box)
        closest_obs = result_box[result_box[:, -1].argsort()][0].tolist()
        return closest_obs

class ENV(object):
    R = -0.1
    PUNISH = -1
    STEP_P = -0.001
    W1, W2, W3 = 5, 150, 80
    SIGMA = 0.3
    obstacle_bound = 0.06

    def __init__(self, max_episode_steps=1e3, GUI=False, num_obstacles=1, kuka=False):
        if GUI: physicsClient = p.connect(p.GUI)
        else: physicsClisent = p.connect(p.DIRECT)  # to connect graphical version; could change to directly
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally, to get pybullet internal data like plane...
        p.setGravity(0, 0, -9.8)
        planeId = p.loadURDF('plane.urdf')

        self.ur5 = UR5_whole()
        self.target = Targetbody('cube_small.urdf', [[0.5, 0.5, 0.37], [0, 0, 0, 1]])
        # self.target = Targetbody('/lego/lego.urdf', [[0.5, 0.5, 0.37], [0, 0, 0, 1]], scale=1)
        # self.target = Targetbody('/objects/mug.urdf', [[0.5, 0.5, 0.37], [0, 0, 0, 1]], scale=0.7)
        self.obstacle = Obstaclebody('soccerball.urdf', [[0.5, 0.5, 0.58], [0, 0, 0, 1]])
        # self.obstacle = Obstaclebody('teddy_vhacd.urdf', [[0.5, 0.5, 0.58], [0, 0, 0, 1]], scale=2)

        self.num_obstacles = num_obstacles
        self.conveyor = Conveyor([[-0.1, 0.5, 0.38], p.getQuaternionFromEuler([0, 0, 0])])
        self.camera = Camera()
        self.camera.load_Matrix()
        # self.all_id = [planeId, self.conveyor.id, self.obstacle.id] # for rl model
        self.all_id = [planeId, self.obstacle.id]
        self._max_episode_steps = max_episode_steps

        # grasp limitation
        self.grasp_path = "D:\Learning\\Newlearning\COMS6998\dynamic-grasping\src\dynamic_grasping_pybullet\\assets\grasps\\filtered_grasps_noise_robotiq_100"
        self.type = "\cube"
        self.one_grasp = np.load(self.grasp_path+self.type+'\\actual_grasps.npy')[19]
        self.one_grasp = [self.one_grasp[:3], self.one_grasp[3:]]

    def reset(self):
        self.conveyor.reset(); self.obstacle.reset()
        # dist = np.random.uniform(0.5, 0.7) # random env

        dist = np.random.uniform(-0.5, -0.7) # random env # 0~np.pi/2 during simulation
        theta = np.random.uniform(-np.pi, np.pi)
        length_obs = np.random.uniform(1, 2)
        length_obj = np.random.uniform(0.8, 1.5)
        self.direction = np.random.choice([-1, 1])
        speed_obj = np.random.uniform(0.5, 0.8)
        type_move = np.random.choice([0, 1, 2, 3, 4, 5])

        if type_move == 0:
            self.conveyor.initialize_linear_motion(dist=dist, theta=theta, length=length_obj, direction=self.direction, speed=speed_obj, z_end=self.conveyor.initial_pose[0][2], variable_speed=False)  # same height
            self.obstacle.initialize_linear_motion(dist=dist, theta=theta, length=length_obj, direction=self.direction, speed=speed_obj, z_end=self.obstacle.initial_pose[0][2], variable_speed=False, is_random=True)  # theta start position
        if type_move == 1:
            self.conveyor.initialize_circular_motion(dist=dist, theta=theta, length=length_obj, speed=speed_obj, direction=self.direction)
            self.obstacle.initialize_circular_motion(dist=dist, theta=theta, length=length_obj, speed=speed_obj, direction=self.direction)
        if type_move == 2:
            self.conveyor.initialize_linear_motion(dist=dist, theta=theta, length=length_obj, direction=self.direction, speed=speed_obj, z_end=self.conveyor.initial_pose[0][2], variable_speed=False)  # same height
            self.obstacle.initialize_circular_motion(dist=dist, theta=theta, length=length_obj*length_obs, speed=speed_obj*length_obs, direction=self.direction)
        if type_move == 3:
            self.conveyor.initialize_sinusoid_motion(dist=dist, theta=theta, length=0.8*length_obj, direction=-1*np.abs(self.direction), speed=0.8*speed_obj)
            self.obstacle.initialize_linear_motion(dist=dist, theta=theta, length=length_obj*length_obs, speed=speed_obj*length_obs, direction=self.direction, z_end=self.obstacle.initial_pose[0][2], variable_speed=False, is_random=False)
        if type_move == 4:
            self.conveyor.initialize_circular_motion(dist=dist, theta=theta, length=length_obj, speed=speed_obj, direction=self.direction)
            self.obstacle.initialize_sinusoid_motion(dist=dist, theta=theta, length=length_obj * length_obs,
                                                     direction=self.direction, speed=speed_obj * length_obs)
        if type_move == 5:
            self.conveyor.initialize_sinusoid_motion(dist=dist, theta=theta, length=0.8*length_obj, direction=-1*np.abs(self.direction), speed=0.8*speed_obj)
            self.obstacle.initialize_sinusoid_motion(dist=dist, theta=theta, length=length_obj * length_obs, direction=-1*np.abs(self.direction), speed=speed_obj * length_obs)

        self.target.puton_conveyor(self.conveyor); self.ur5.reset()
        p.removeAllUserDebugItems(); steps(100)
        self.future_steps = 60
        return self.query_state()

    def random_action(self):
        action = np.random.uniform(-np.pi, np.pi, size=6)
        return action

    def tip2target(self):
        Tip_pos = self.ur5.endeffector2gripper()[0] # tip-center
        Target_pos = get_pose(self.target.id)[0]
        Distance = distance(Tip_pos, Target_pos)
        return Distance

    def reward_function(self, action): # function design based on env
        # one reward to control orientation: 3 points in one line
        reward = 0; terminal = False
        Distance = self.tip2target()
        if Distance <= self.SIGMA: reward += -0.5 * Distance ** 2 * self.W3 # Euclidean reward
        else: reward += -self.SIGMA * (Distance - 0.5 * self.SIGMA) * self.W3

        # if self.FLAG: reward += self.PUNISH * self.W2 # bad joint angle output
        if self.ur5.collision_check(self.all_id): reward += self.PUNISH * self.W2; terminal = True# get collision
        if self.conveyor.check_done(): reward += self.PUNISH * self.W2/10; terminal = True # object stops movement

        if Distance <= 0.02: reward = self.R; terminal = True # get touch reward
        reward += -self.W1 * np.linalg.norm(action) # restrict action
        reward += self.STEP_P
        return reward, terminal

    def query_state(self): # get current state, needs to change
        _, _, _ = self.camera.getCameraImage()
        state_ur5 = self.ur5.get_arm_joint_values()
        # state_gripper = list(self.ur5.endeffector2gripper()[0]) + list(self.ur5.endeffector2gripper()[1])
        state_gripper = list(self.ur5.endeffector2gripper()[0])
        state_target = get_pose(self.target.id)
        # TODO get all obstacles with their parameters: current/future pos, orient, extension
        future_index_obs = self.obstacle.wp_target_index+self.future_steps if self.obstacle.wp_target_index+self.future_steps < len(self.obstacle.discretized_trajectory) else len(self.obstacle.discretized_trajectory)-1
        obstacle_box = [[self.obstacle.get_pose()[0], self.obstacle.get_pose()[1], [self.obstacle_bound, self.obstacle_bound, self.obstacle_bound]],
                         [self.obstacle.discretized_trajectory[future_index_obs][0], self.obstacle.discretized_trajectory[future_index_obs][1], [self.obstacle_bound, self.obstacle_bound, self.obstacle_bound]]]
        # obstacle_box = [[self.obstacle.get_pose()[0], self.obstacle.get_pose()[1], [self.obstacle_bound, self.obstacle_bound, self.obstacle_bound]]]

        current_obs = self.ur5.get_closest_obs(obstacle_box)
        state = state_target[0] + state_target[1] + current_obs + state_ur5 + state_gripper
        return state

    def step(self, action, model='train'):
        fake_done = False; success = False
        real_pose = np.array(self.ur5.get_arm_joint_values()) + np.array(action) # action is Δq
        # self.FLAG = self.ur5.filter_ik(real_pose, self.all_id) # filter ik whether would collide to obstacle to control its performance
        self.FLAG = False
        if not self.FLAG:
            self.ur5.plan_arm_joint_values(real_pose)
            while not self.ur5.arm_finished():  # move to that pose
                self.ur5.step(); self.conveyor.step(); self.obstacle.step(); steps(1)
        else: self.conveyor.step(); self.obstacle.step(); steps(1)

        if distance(self.ur5.endeffector2gripper()[0], self.target.get_pose()[0]) < 0.15:
            orientation = self.ur5.endeffector2gripper()[1] # directly use current orientation
            joint_angle = self.ur5.ik_pybullet(self.target.get_pose()[0], orientation)
            self.ur5.plan_arm_joint_values(joint_angle, duration=0.12)
            self.ur5.execute_grasp() # dynamic touch no gripper
            # self.ur5.close_gripper()
            # self.ur5.execute_gripper()
            success = self.check_success()
            fake_done = True

        reward, done = self.reward_function(action)
        next_state = self.query_state()
        done = fake_done or done

        self.target.add_line(color=[0, 0, 1])
        self.obstacle.add_line(color=[1, 0, 0])

        if model=='train': return next_state, reward, done
        else: return next_state, success, done

    # rrt part

    def rrt_planner(self, model=0):
        X_dimensions = np.array([(-1, 1), (-1, 1), (0, 1)])
        Q = np.array([(0.01, 1)])  # length of tree edges
        r = 0.01  # length of smallest edge to check for intersection with obstacles
        max_samples = 800  # max number of samples to take before timing out
        prc = 0.1  # probability of checking for a connection to goal # +40 --> 1/6s
        self.future_steps = np.max(self.future_steps-1, 0) # keep reducing
        # TODO get all obstacles with their parameters: current/future pos, orient, extension
        # //use get_all() --> obstacle_box = [[self.obstacle.get_pose()[0], self.obstacle.get_pose()[1], [self.obstacle_bound, self.obstacle_bound, self.obstacle_bound]]]

        future_index_obs = self.obstacle.wp_target_index+self.future_steps if self.obstacle.wp_target_index+self.future_steps < len(self.obstacle.discretized_trajectory) else len(self.obstacle.discretized_trajectory)-1
        future_index_obj = self.conveyor.wp_target_index+self.future_steps if self.conveyor.wp_target_index+self.future_steps < len(self.conveyor.discretized_trajectory) else len(self.conveyor.discretized_trajectory)-1
        obstacle_now = np.array(self.obstacle.get_pose()[0])
        obstacle_future = np.array(self.obstacle.discretized_trajectory[future_index_obs][0])

        # //upper bound vertex, lower bound vertex to replace obstalce_future - obstacle_bound
        obstacles = np.array([tuple(list(obstacle_future - self.obstacle_bound) + list(obstacle_future + self.obstacle_bound)), tuple(list(obstacle_now - self.obstacle_bound) + list(obstacle_now + self.obstacle_bound))]) # become a cube
        # //future or not future!!!!!!!!!!!!!!!!!!!!!
        # obstacles = np.array([tuple(list(obstacle_now - self.obstacle_bound) + list(obstacle_now + self.obstacle_bound))])
        # future_index_obj = self.conveyor.wp_target_index

        X_init = self.ur5.endeffector2gripper()[0]
        if model==0:
            X_goal = self.conveyor.discretized_trajectory[future_index_obj][0].copy() # replace with object future pose
            X_goal[2] += 0.05
        elif model == 1:
            X_goal = self.target.get_pose()[0]
        else: X_goal = self.ur5.HOME_POS # move back to original, still not do
        X_goal = tuple(X_goal)
        X = SearchSpace(X_dimensions, obstacles)
        rrt = RRT(X, Q, X_init, X_goal, max_samples, r, prc)
        self.path = rrt.rrt_search()

    def planner_step(self):
        grasp_model = 0
        if distance(self.ur5.endeffector2gripper()[0], self.target.get_pose()[0]) < 0.1: # closed to object and enough close to future pose, start to grasp
            grasp_model = 1
        duration = None if grasp_model==0 else 0.02
        self.rrt_planner(grasp_model)
        for step in self.path: # a sequence of path position

            # TODO reachability grasp
            # grasp_position, grasp_orientation = convert_grasp_in_object_to_world(self.target.get_pose(), self.one_grasp)
            # partial_joint = self.ur5.ik_wrist(self.target.get_pose()[0], step)
            # control_joints(self.ur5.id, self.ur5.GROUP_INDEX['arm'][:-1], partial_joint)
            # steps(20)
            # p.addUserDebugLine(self.ur5.wrist_pos, self.target.get_pose()[0], lineColorRGB=[1, 0, 0], lifeTime=0.1)
            # joints_angle = p.calculateInverseKinematics(self.ur5.id, self.ur5.EEF_LINK_INDEX, self.ur5.ee_pos)# orientation needs to change, needs to use ik_fastpy with filter

            # angle = -self.direction * calculate_angle(self.target.get_pose()[0])  # get orientation for current object pos
            angle = 0
            joints_angle = self.ur5.ik_pybullet(step, p.getQuaternionFromEuler([3*np.pi/4, 0, angle]))
            self.ur5.plan_arm_joint_values(joints_angle, duration=duration)
            self.ur5.step()
            # self.ur5.add_line()
            self.conveyor.step()
            self.obstacle.step()
            if grasp_model==0: steps(2)
            elif grasp_model==1: self.ur5.execute_grasp() # direct go to grasp
        if grasp_model==1: # close gripper
            self.ur5.close_gripper()
            self.ur5.execute_gripper()
            return self.check_success()

        if self.path == []: # have inserted in obstacles, manually move out
            self.ur5.plan_arm_joint_values(self.ur5.HOME, duration=0.05)
            self.ur5.step(); steps(1)
        self.target.add_line(color=[0, 0, 1])
        self.obstacle.add_line(color=[1, 0, 0])
        steps(1)

    def check_success(self):
        steps(1)
        if distance(self.ur5.endeffector2gripper()[0], self.target.get_pose()[0]) < 0.05:
            print('Sucess!')
            return 1
        print('Fail!')
        return 0

    def close(self):
        p.disconnect()

def distance(x, y):
    x, y = np.array(x), np.array(y)
    return np.sqrt(np.mean(np.square(x-y)))

def closest(id1, id2, threshold=1): # only distance smaller than 1 would be returned
    all_points = p.getClosestPoints(id1, id2, threshold)
    buffer = []
    for point in all_points:
        buffer.append(point[8])
    if len(buffer) > 0: return min(buffer)
    else: return None

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat)

def get_pose(body):
    raw = p.getBasePositionAndOrientation(body)
    position = list(raw[0])
    orn = list(raw[1])
    return [position, orn]

def get_joint_info(body, joint): # ???
    return JointInfo(*p.getJointInfo(body, joint))
def get_body_info(body):
    return BodyInfo(*p.getBodyInfo(body))

BASE_LINK = -1
def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')
def get_link_name(body, link):
    if link == BASE_LINK:
        return get_base_name(body)
    return get_joint_info(body, link).linkName.decode('UTF-8')
def link_from_name(body, name):
    if name == get_base_name(body):
        return BASE_LINK
    for link in range(p.getNumJoints(body)):
        if get_link_name(body, link) == name:
            return link
    raise ValueError(body, name)

def wrap_angle(theta): # ???
    return (theta + np.pi) % (2 * np.pi) - np.pi
def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)
def is_circular(body, joint):
    joint_info = get_joint_info(body, joint)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit # ???
def get_difference_fn(body, joints):
    def fn(q2, q1):
        difference = []
        for joint, value2, value1 in zip(joints, q2, q1):
            difference.append(circular_difference(value2, value1)
                              if is_circular(body, joint) else (value2 - value1))
        return list(difference)
    return fn

def get_max_velocity(body, joint):
    return get_joint_info(body, joint).jointMaxVelocity
def get_max_force(body, joint):
    print(get_joint_info(body, joint).jointMaxForce)
    return get_joint_info(body, joint).jointMaxForce

def set_joint_position(body, joint, value):
    p.resetJointState(body, joint, value)
def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value)

def control_joint(body, joint, value):
    return p.setJointMotorControl2(bodyUniqueId=body,
                                   jointIndex=joint,
                                   controlMode=p.POSITION_CONTROL,
                                   targetPosition=value,
                                   targetVelocity=0,
                                   maxVelocity=get_max_velocity(body, joint),
                                   force=get_max_force(body, joint))
def control_joints(body, joints, positions):
    return p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL,
                                       targetPositions=positions,
                                       targetVelocities=[0.0] * len(joints),
                                       forces=[10000] * len(joints)) #forces=[get_max_force(body, joint) for joint in joints]

def forward_kinematics(body, joints, positions, eef_link=None):
    eef_link = get_num_joints(body) - 1 if eef_link is None else eef_link
    old_positions = get_joint_positions(body, joints)
    set_joint_positions(body, joints, positions)
    eef_pose = get_link_pose(body, eef_link)
    set_joint_positions(body, joints, old_positions)
    return eef_pose

def calculate_angle(pos):
    angle = np.arctan2(pos[1], pos[0])
    return angle

def create_cube(pos, orient, extension=[0.5, 0.5, 0.5]):
    cubeid = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=extension)
    mass = 0
    bodyid = p.createMultiBody(mass, cubeid, basePosition=pos, baseOrientation=orient)
    return cubeid, bodyid

def del_cube(cubeid, bodyid):
    p.removeBody(bodyid)

def convert_grasp_in_object_to_world(object_pose, grasp_in_object):
    """
    :param object_pose: 2d list
    :param grasp_in_object: 2d list
    """
    object_position, object_orientation = object_pose
    grasp_position, grasp_orientation = grasp_in_object
    grasp_in_world = p.multiplyTransforms(object_position, object_orientation, grasp_position, grasp_orientation)
    return grasp_in_world

def steps(n):
    for i in range(n): p.stepSimulation()
    time.sleep(1/240)
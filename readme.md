# Adaptive Grasp/Touch in Dynamic Environment(UR5)
Currently working & updating

## Introduction
When working in static environments, robot usually gets outperform and stable performance. However, robotic manipulation is much harder in dynamic environments or in real world. For instance, in dynamic grasping, moving balls catch, the targets and obstacles might be moving with an unknown motion. Therefore, the ability of manipulating moved targets while facing moving obstacles is necessary for a robot in both industry and daily life. Previous works either mainly focus on grasping moving objects [1-2], avoiding static obstacles [3-4], grasping static object with moving obstacles [5]. In this project, we relax some of these assumptions and try to solve the problem of robotic grasping for moving objects with moving obstacles by using one camera observation to predict obstacles and object future pose. To sum up, our currently main contributions are:<br>
***(1) Obstacles point cloud and object position prediction: using one camera history observations to predict their future pose.<br>
(2) A reinforcement-learning motion planner: directly input observation and output joint actions to control robot arm without inverse kinematics.<br>
(3) Simulation robot evaluation: procedure of systematically evaluating dynamic grasping/touching performance in a simulation environment with randomized linear / sinusoid / circular.***<br>

`Our current framework is`:
<div align=center>
<img src='https://github.com/Ys-Jia/Adaptive-Grasp-in-Dynamic-Environment/blob/main/Framework.png' width="600" height="600">
</div>
We first use graspit to generate analytical grasp pose for a certain object Graw, then we use FilterGraspDataBase function to filter grasp based on our simulation physical environment to output certain numbers of filter grasps with highest success rates GDB. We randomly pick one as this time’s grasp pose. Then we record one RGB-D picture O based on CameraRecordRGBD and use it to get two objects pose Pobj which contains current pose and next Tfuture steps pose, and two obstacles point cloud Pobs. Then we use our MotionPlanner which takes Pobj, Pobs, GDB and current robot configuration q to output joint position control signal Δq, moving speed. We use ForwardCheck to update robotic arm current configuration q and do collision check. If Collision happens, we would reset the environment and record failure. Otherwise, if the distance between the end-effector position f(q) with the current object position Pobj-cur is smaller than a threshold δ, we would execute Grandom, move to the object, and close the gripper. This function finally returns the success flag for evaluation.

## Visualization(Pybullet)
Please make sure all files in `Code_Package` are in your local directory(we would add requirements.txt soon)!<br>

Here are samples of dynamic grasp, the first three are RRT planner and the last one is SAC planner:
<div align=center>
<img src='https://github.com/Ys-Jia/Adaptive-Grasp-in-Dynamic-Environment/blob/main/RRT-Circle.gif' width="400" height="300"> 
<img src='https://github.com/Ys-Jia/Adaptive-Grasp-in-Dynamic-Environment/blob/main/RRT-Combined.gif' width="400" height="300">  
</div>
<div align=center>
<img src='https://github.com/Ys-Jia/Adaptive-Grasp-in-Dynamic-Environment/blob/main/RRT-Sin.gif' width="400" height="300"> 
<img src='https://github.com/Ys-Jia/Adaptive-Grasp-in-Dynamic-Environment/blob/main/SAC-success-original%20speed.gif' width="400" height="300">  
</div>

## Current Results
Perception Performance:<br>
<div align=center>
<img src='https://github.com/Ys-Jia/Adaptive-Grasp-in-Dynamic-Environment/blob/main/Perception-Table.png' width="800" height="300">  
</div>
Grasp/Touch Performance:<br>
<div align=center>
<img src='https://github.com/Ys-Jia/Adaptive-Grasp-in-Dynamic-Environment/blob/main/Motion-Planning-Table-RRT.png' width="1200" height="200">  
<img src='https://github.com/Ys-Jia/Adaptive-Grasp-in-Dynamic-Environment/blob/main/Motion-Planning-Table-SAC.png' width="1200" height="200"> 
</div>

## Reference
>[1] Iretiayo Akinola, Jingxi Xu, Shuran Song, Peter K. Allen, “Dynamic Grasping with Reachability and Motion Awareness,” CoRR abs/2103.10562(2021)<br>
>[2] D. Morrison, P. Corke, and J. Leitner, “Closing the loop for robotic grasping: A real-time, generative grasp synthesis approach,” 2018.<br>
>[3] T. Zhang, K. Zhang, J. Lin, W. -Y. G. Louie and H. Huang, ”Sim2real Learning of Obstacle Avoidance for Robotic Manipulators in Uncertain Environments,” in IEEE Robotics and Automation Letters, vol. 7, no. 1, pp. 65-72, Jan. 2022.<br>
>[4] H. Xie, G. Li, Y. Wang, Z. Fu, and F. Zhou, “Research on visual servo grasping of household objects for nonholonomic mobile manipulator,” Journal of Control Science and Engineering, vol. 2014, p. 16, 2014.<br>
>[5] D. Kappler, F. Meier, J. Issac, J. Mainprice, C. G. Cifuentes, M. W¨uthrich, V. Berenz, S. Schaal, N. Ratliff, and J. Bohg, “Realtime perception meets reactive motion generation,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 1864–1871, 2018.<br>

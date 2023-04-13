---
title: Franka Panda Arm with ROS
date: 2023-04-13 17:23:00 -0500
categories: [Tutorial]
tags: [ros_noetic, panda_arm]
description: This tutorial presents a step-by-step guide to set up Franka Emika Panda Arm with ROS Noetic and create ROS packages to operate it for pick and place tasks.
seo:
  date_modified: 2023-04-13 17:23:59 -0500
---

The Franka Emika Panda Arm is a seven-axis industrial robot arm that is becoming increasingly popular in robotics research and manufacturing. It is easy to use, has a high payload capacity, and comes with an integrated controller. This tutorial will guide you through the process of installing ROS Noetic and all necessary packages, connecting the robot to your workstation, and controlling it using ROS.

## 1. Install ROS Noetic and all necessary packages on your Workstation PC
Before we can start using the Franka Emika Panda Arm, we need to install ROS Noetic and all the necessary packages. We will be using MoveIt! for motion planning and control. Follow the steps below to install ROS Noetic and MoveIt!.

- Follow this [documentation](https://ros-planning.github.io/moveit_tutorials/doc/getting_started/getting_started.html){:target="_blank"} and install all the necessary packages including ROS Noetic, Catkin, and MoveIt packages.
- **Note**: Make sure you're ready with your **ws_moveit** workspace with all the packages downloaded and installed by the end of the above step.


## 2. Install a Real-time Kernel to enable operation of the robot
To work with robots, it is a good practice to install real-time kernels since they are configured for low latency tasks. The Franka Emika Panda Arm requires a real-time kernel to function properly. Follow the steps below to install the real-time kernel.

- Go to this documentation link for the whole process: [real-time kernel installation](https://frankaemika.github.io/docs/installation_linux.html#setting-up-the-real-time-kernel){:target="_blank"}.
- Once you install the kernel, verify it by rebooting the PC and selecting the kernel from GRUB boot options.
  - If for any reason, you don't see a GRUB menu while booting up, keep pressing `Shift+Space` when the PC turns on. It should display the GRUB options.
- **Note**: If you've any issues with NVIDIA drivers when installing a Realtime kernel, refer to this: [nvidia-driver-installation](https://gist.github.com/pantor/9786c41c03a97bca7a52aa0a72fa9387){:target="_blank"}


## 3. Connect the robot arm to your Workstation PC
Now that we have installed ROS Noetic and a real-time kernel, we can connect the robot to our workstation. Follow the steps below to connect the robot.

- Use an ethernet cable to connect to the Shop Floor Controller (big box).
  - Use a router or a switch to connect the Controller box and your Workstation PC.
- Note down the IP address of the shop controller (you can use linux tools like nmap or just log in to the router to find it out).
  - `nmap -sn <router_ip>/24`
- Another way to find out the shop controller's IP is by using the following command:
```
nmap -sP <router_ip>/24 | grep -B 2 "4C:52:62:0F:1C:4C" | head -n 1 | cut -d " " -f 5
```
  - First, check the router's host IP and run the following command:
  - For the current setup, <router_ip> is 192.168.88.1 and the IP of the robot control is 192.168.88.10.


## 4. Franka Desk Initial Setup
Before you can connect to the Franka robot, you need to first connect to the Franka Desk using the following steps:

- Before you turn the robot ON, first make sure that the **Activation Lock switch (black) is turned OFF**. This keeps the robot locked and prevents from external manual operation of the robot.
- Once the robot is turned ON, follow the following steps:
  - Enter the IP address of the robot controller (e.g. 192.168.88.10) that you figured our in step 3 on a web browser.
  - Log in with the following information:
  ```
  Username: <username>
  Password: <password>
  ```
- First, unlock the **Joints** from the Franka Desk.
- Next, **Activate FCI** from the dropdown menu on the Desk page. This will enable communication between the PC and the robot.
- Now, you're ready to launch ros nodes and operate the robot.


## 5. Connect to the robot
Make sure that you complete all the previous steps before moving to this one as the packages need to be properly installed to connect to the robot.


- Once you've downloaded all the packages in the **ws_moveit** workspace from step #1.
- Run the following command to launch a roscore and connect to the robot:
  ```
  roslaunch panda_moveit_config franka_control.launch robot_ip:=<robot_IP> load_gripper:=true
  ```
- `<robot_IP>` in this case is: 192.168.88.10
- This should start a connection with the robot and you'll be able to see it's current state on RViz.

## 6. (Optional) Make the robot move
To make the robot move, you can use one of the MoveIt! tutorials to move the robot and verify that it is connected and working. Follow these steps:

- Make sure that there's enough space for the robot to move in all directions.
- You can use one of the moveit tutorials to move and the robot and verify whether your robot is connected and working.
- Run the following command in a new terminal:
  - `rosrun moveit_tutorials move_group_python_interface_tutorial.py`
  - This node will move the robot to a particular hardcoded location.

## 7. Create your own package in the workspace and write your own node.
Now, it's time to create your own ROS package and write your first custom node. Follow these steps to get started:


- Inside `/path/to/ws_moveit/src`, create your custom package:
  - `catkin_create_pkg custom_panda_arm rospy`
- Create a node inside the package:
  - `touch custom_panda_arm/src/pick_place.py`
  - Add `#!/usr/bin/env python3` to the top of the python file and make it executable using:
    - `chmod +x custom_panda_arm/src/pick_place.py`
- Now, you can start writing your node to control the arm. Please refer to this [example](https://gist.github.com/asheeshcric/6e40e98780919876cac2d7657e1977c6)
  - This script moves the arm to a hard-coded location. You may change the positions and orientation accordingly, making sure that the robot doesn't self-collide.
- Finally, run the new node using: `rosrun custom_panda_arm pick_place.py`


## Franka Lights Meaning

- **Yellow Lights Blinking**: Booting
- **Blue Lights**: Ready to Use
- **White Lights**: Ready to use, but the joints have to be unlocked from the panda's interface (Desk).
- **Red lights**: The robot has encountered some potential illegal moves which can cause the robot to have a self-collision. It's better to shut down and reboot the robot.
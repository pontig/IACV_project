# 🛰️ Drone Trajectory Reconstruction from Multiple Unsynchronized Cameras

This project implements a computer vision pipeline to reconstruct the 3D trajectory of a UAV (drone) using videos from multiple unsynchronized and uncalibrated cameras. The work is inspired by the method proposed by Li et al., and is built using Python, OpenCV, and ROS2.

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Camera Intrinsic Calibration](#1-camera-intrinsic-calibration)
  - [2. Drone Detection](#2-drone-detection)
  - [3. Offline Trajectory Reconstruction](#3-offline-trajectory-reconstruction)
  - [4. Online Trajectory Reconstruction (ROS2)](#4-online-trajectory-reconstruction-ros2)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Credits](#credits)

---

## 🧠 Project Overview

This project reconstructs a drone's 3D flight path using external video footage without requiring synchronized or pre-calibrated cameras. It provides:

- Intrinsic camera calibration
- Multiple drone detection methods
- Offline and online 3D trajectory reconstruction
- Modular code structure for extendability

## 🔧 Features

- ✔️ Camera intrinsic calibration using chessboard patterns
- ✔️ Drone detection using:
  - Background subtraction + optical flow
  - YOLOv8 object detection
- ✔️ Offline 3D trajectory reconstruction with bundle adjustment
- ✔️ Online trajectory reconstruction in real-time with ROS2
- ❌ No rolling shutter correction or motion regularization (yet)

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your_username/drone-trajectory-reconstruction.git
cd drone-trajectory-reconstruction
pip install -r requirements.txt
```

Ensure the dataset is cloned into the root directory:

```bash
git clone https://github.com/CenekAlbl/drone-tracking-datasets.git
```

Clean up calibration file names as described in the documentation.

---

## 🚀 Usage

### 1. Camera Intrinsic Calibration

```bash
python camera_calib.py
```

Saves intrinsic parameters and comparison plots.

---

### 2. Drone Detection

**Classical (Background Subtraction + Optical Flow):**

```bash
python moving_video.py --y_limit 1080 <dataset> <camera>
```

**YOLOv8:**

```bash
python yolo_detection.py --model yolov8n.pt --save_video <dataset> <camera>
```

---

### 3. Offline Trajectory Reconstruction

#### Step 1: Estimate temporal shift (β search):

```bash
python beta_search.py <dataset_no>
```

#### Step 2: Run reconstruction:

```bash
python main.py <dataset_no>
```

---

### 4. Online Trajectory Reconstruction (ROS2)

#### Setup:

```bash
cd ros_node
colcon build
source install/setup.bash
```

#### Run the main reconstruction node:

```bash
ros2 run drones drones
```

#### Replay detections from bag file:

```bash
ros2 bag play bags/dataset4.bag -r 1.5
```

To regenerate the bag from CSV:

```bash
ros2 run drones live_detections_generator
ros2 bag record -o <name> /detections
```

Edit `drones.py` to set dataset number and algorithm parameters.

---

## 📁 Repository Structure

```
📁 documentation/          # Project report and ppt presentation
📁 homework/               # The repository of IACV mid homework
📁 plots/                  # All plots, videos, and evaluation results
📁 ros_node/               # ROS2 package for online tracking
│   ├── drones/            # Main ROS2 nodes
│   ├── bags/              # Sample ROS2 bags
│   └── config/            # RViz2 configs
📄 camera_calib.py         # Intrinsic calibration
📄 moving_video.py         # Classical drone detection
📄 yolo_detection.py       # YOLOv8-based drone detection
📄 beta_search.py          # Temporal alignment
📄 main.py                 # Offline reconstruction
📄 requirements.txt
```

Other files are for internal use, such as generating plots and images for the report.

---


## 👨‍🔬 Credits

Developed by **Elia Pontiggia**  
Project for *Image Analysis and Computer Vision (2025)*  
Based on the paper by [Li et al.](https://arxiv.org/abs/2003.04784) and dataset by [Cenek Albl](https://github.com/CenekAlbl/drone-tracking-datasets)

**Evaluation**: this project scored 30/30 in the IACV course.

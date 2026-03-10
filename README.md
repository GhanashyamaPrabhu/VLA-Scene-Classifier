# VLA Autonomous Vehicle Scene Classifier

Real-time scene classification and context-aware perception for autonomous vehicles using Vision-Language AI and an Orbbec Femto Bolt depth camera.

## Hardware
- Orbbec Femto Bolt (RGB + Depth + IR + IMU)
- Ubuntu 22.04
- ROS 2 Humble

## Features
- Live RGB + Depth stream via ROS 2
- Scene classification using Claude Vision API
- Depth-aware spatial context (Left/Center/Right zones)
- Real-time OpenCV overlay

## Setup
```bash
ros2 launch orbbec_camera femto_bolt.launch.py
python3 scene_classifier.py
```

## Output
SCENE / HAZARDS / ROAD / TRAFFIC / VISIBILITY / ACTION

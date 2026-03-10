# 🚗 VLA Autonomous Vehicle Scene Classifier

Real-time scene classification and context-aware perception for autonomous vehicles using Vision-Language AI, YOLOv8 object detection, and an Orbbec Femto Bolt RGB-D camera.

## 📷 Hardware
- **Camera:** Orbbec Femto Bolt (RGB + Depth + IR + IMU)
- **Workstation:** HP Z440
- **OS:** Ubuntu 22.04 LTS
- **Connection:** USB 3.1

## 🧠 AI Stack
- **Scene Understanding:** Anthropic Claude Vision API (claude-opus-4-5)
- **Object Detection:** YOLOv8 Nano (real-time, local)
- **Depth Analysis:** Orbbec Femto Bolt depth stream (3-zone spatial context)

## ⚙️ System Stack
- ROS 2 Humble Hawksbill
- OrbbecSDK ROS2 Driver
- Python 3.10
- OpenCV

## 🎯 Features
- ✅ Live RGB + Depth streaming via ROS 2
- ✅ Real-time YOLOv8 object detection with bounding boxes
- ✅ Scene classification every ~1 second via Claude Vision API
- ✅ Depth-aware spatial context (Left / Center / Right obstacle zones)
- ✅ Color-coded live overlay on camera feed
- ✅ Results published as ROS 2 topic (/scene_classification)
- ✅ Video mode for offline testing
- ✅ Async inference (camera feed never freezes)

## 📊 Classification Output
```
SCENE:      urban / highway / suburban / parking / intersection / tunnel
HAZARDS:    pedestrians, vehicles, obstacles, or none
ROAD:       dry / wet / icy / construction / unknown
TRAFFIC:    none / light / moderate / heavy / stopped
VISIBILITY: good / moderate / poor
ACTION:     recommended driving action
```

## 🎨 YOLO Bounding Box Colors
| Color | Object |
|-------|--------|
| 🔴 Red | Pedestrians, Stop Signs |
| 🟢 Green | Cars |
| 🟠 Orange | Trucks, Buses |
| 🔵 Blue | Bicycles, Motorcycles |
| 🔵 Cyan | Traffic Lights |

## 🚀 Installation

### 1. ROS 2 Humble
```bash
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### 2. Orbbec ROS2 Driver
```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/orbbec/OrbbecSDK_ROS2.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
```

### 3. Python Dependencies
```bash
pip3 install "numpy<2" "opencv-python<4.9" anthropic Pillow ultralytics
sudo apt install -y ros-humble-cv-bridge python3-cv-bridge
```

### 4. API Key
```bash
nano ~/.bashrc
# Add: export ANTHROPIC_API_KEY=your-key-here
source ~/.bashrc
```

## ▶️ How to Run

### Live Camera Mode
```bash
# Terminal 1 - Start camera
ros2 launch orbbec_camera femto_bolt.launch.py

# Terminal 2 - Start classifier
python3 scene_classifier.py
```

### Video Mode (Offline Testing)
```bash
python3 video_classifier.py path/to/video.mp4
```

### Controls (Video Mode)
- **SPACE** — Pause / Resume
- **Q** — Quit

## 📡 ROS 2 Topics
| Topic | Type | Description |
|-------|------|-------------|
| /camera/color/image_raw | sensor_msgs/Image | RGB feed |
| /camera/depth/image_raw | sensor_msgs/Image | Depth feed |
| /camera/gyro_accel/sample | sensor_msgs/Imu | IMU data |
| /scene_classification | std_msgs/String | VLA output |

## 🔮 Roadmap
- [ ] Lane detection with OpenCV
- [ ] Nav2 path planner integration
- [ ] IMU-based motion detection
- [ ] Multi-camera surround view
- [ ] Local VLA model (RTX 3060+)
- [ ] Fine-tuning on custom driving data
- [ ] Web dashboard for remote monitoring

## 📁 Project Structure
```
av_vla_project/
├── scene_classifier.py    # Live camera mode (ROS 2)
├── video_classifier.py    # Video mode (offline testing)
└── README.md
```

## 👤 Author
**GhanashyamaPrabhu**
GitHub: https://github.com/GhanashyamaPrabhu

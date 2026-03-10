#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import anthropic
import base64
import threading
import time
from PIL import Image as PILImage
import io
from ultralytics import YOLO

class AVSceneClassifier(Node):
    def __init__(self):
        super().__init__('av_scene_classifier')
        self.bridge = CvBridge()
        self.client = anthropic.Anthropic()

        # Load YOLO model
        self.get_logger().info('Loading YOLOv8 model...')
        self.yolo = YOLO('yolov8n.pt')  # nano = fastest
        self.get_logger().info('YOLOv8 ready!')

        # State
        self.latest_rgb = None
        self.latest_depth = None
        self.last_result = "Initializing scene classifier..."
        self.last_detections = []
        self.classifying = False
        self.frame_count = 0
        self.CLASSIFY_EVERY = 30

        # ROS2 Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # ROS2 Publisher for scene results
        self.scene_pub = self.create_publisher(String, '/scene_classification', 10)

        self.get_logger().info('AV Scene Classifier with YOLO started!')

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_count += 1

        # Run YOLO on every frame (fast, local)
        self.run_yolo(self.latest_rgb)

        # Run Claude API every N frames (slower, cloud)
        if self.frame_count % self.CLASSIFY_EVERY == 0 and not self.classifying:
            thread = threading.Thread(target=self.classify_scene)
            thread.daemon = True
            thread.start()

        self.display_frame()

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')

    def run_yolo(self, frame):
        results = self.yolo(frame, verbose=False, conf=0.4)
        self.last_detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo.names[cls]
                self.last_detections.append({
                    'label': label,
                    'conf': conf,
                    'box': (x1, y1, x2, y2)
                })

    def get_depth_context(self):
        if self.latest_depth is None:
            return "Depth data unavailable"
        depth = self.latest_depth.astype(float)
        h, w = depth.shape
        zones = {
            "left":   depth[:, :w//3],
            "center": depth[:, w//3:2*w//3],
            "right":  depth[:, 2*w//3:]
        }
        context = []
        for zone, data in zones.items():
            valid = data[(data > 200) & (data < 8000)]
            if len(valid) > 0:
                min_d = np.min(valid) / 1000.0
                context.append(f"{zone}: {min_d:.1f}m")
        return "Nearest obstacles - " + ", ".join(context) if context else "No obstacles"

    def get_yolo_summary(self):
        if not self.last_detections:
            return "No objects detected"
        counts = {}
        for d in self.last_detections:
            counts[d['label']] = counts.get(d['label'], 0) + 1
        return ", ".join([f"{v}x {k}" for k, v in counts.items()])

    def classify_scene(self):
        if self.latest_rgb is None:
            return
        self.classifying = True
        try:
            frame = cv2.resize(self.latest_rgb, (640, 360))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb_frame)

            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            img_b64 = base64.standard_b64encode(
                buffer.getvalue()).decode('utf-8')

            depth_context = self.get_depth_context()
            yolo_context = self.get_yolo_summary()

            prompt = f"""You are an autonomous vehicle perception system. Analyze this camera frame and respond in this exact format:

SCENE: [urban/highway/suburban/parking/intersection/tunnel/residential]
HAZARDS: [list any immediate dangers or none]
ROAD: [dry/wet/icy/construction/unknown]
TRAFFIC: [none/light/moderate/heavy/stopped]
VISIBILITY: [good/moderate/poor]
ACTION: [recommended driving action in one sentence]

Depth sensor data: {depth_context}
YOLO detections: {yolo_context}
Be concise and accurate."""

            response = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            self.last_result = response.content[0].text
            self.get_logger().info(
                f"\n{'='*50}\n{self.last_result}\n{'='*50}")

            # Publish to ROS2 topic
            msg = String()
            msg.data = self.last_result
            self.scene_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Classification error: {str(e)}")
            self.last_result = f"Error: {str(e)}"
        finally:
            self.classifying = False

    def draw_yolo_boxes(self, frame):
        # Color map for common AV-relevant classes
        color_map = {
            'person':     (0, 0, 255),    # Red
            'car':        (0, 255, 0),    # Green
            'truck':      (0, 165, 255),  # Orange
            'bus':        (0, 165, 255),  # Orange
            'bicycle':    (255, 0, 0),    # Blue
            'motorcycle': (255, 0, 0),    # Blue
            'traffic light': (0, 255, 255),  # Cyan
            'stop sign':  (0, 0, 255),    # Red
        }
        for det in self.last_detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            conf = det['conf']
            color = color_map.get(label, (200, 200, 200))

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-6),
                         (x1+tw+4, y1), color, -1)
            cv2.putText(frame, text, (x1+2, y1-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def display_frame(self):
        if self.latest_rgb is None:
            return
        display = self.latest_rgb.copy()

        # Draw YOLO bounding boxes
        display = self.draw_yolo_boxes(display)

        # Dark overlay for scene text
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (700, 185), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        # Draw scene classification results
        y = 25
        for line in self.last_result.strip().split('\n'):
            if line.strip():
                color = (0, 255, 0)
                if 'HAZARD' in line and 'none' not in line.lower():
                    color = (0, 0, 255)
                elif 'ACTION' in line:
                    color = (0, 255, 255)
                cv2.putText(display, line.strip(), (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y += 26

        # Object count bar
        obj_text = f"YOLO: {self.get_yolo_summary()}"
        cv2.putText(display, obj_text,
                   (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                   (255, 255, 0), 1)

        # Status indicator
        status = "ANALYZING..." if self.classifying else "LIVE"
        color = (0, 165, 255) if self.classifying else (0, 255, 0)
        cv2.putText(display, status,
                   (display.shape[1]-130, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("AV Scene Classifier - Femto Bolt", display)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = AVSceneClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import anthropic
import base64
import threading
import time
from PIL import Image as PILImage
import io

class AVSceneClassifier(Node):
    def __init__(self):
        super().__init__('av_scene_classifier')
        self.bridge = CvBridge()
        self.client = anthropic.Anthropic()
        
        # State
        self.latest_rgb = None
        self.latest_depth = None
        self.last_result = "Initializing scene classifier..."
        self.classifying = False
        self.frame_count = 0
        self.CLASSIFY_EVERY = 30  # every 30 frames (~1 sec)

        # ROS2 Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        self.get_logger().info('AV Scene Classifier started!')

    def rgb_callback(self, msg):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_count += 1
        if self.frame_count % self.CLASSIFY_EVERY == 0 and not self.classifying:
            thread = threading.Thread(target=self.classify_scene)
            thread.daemon = True
            thread.start()
        self.display_frame()

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

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
        return "Nearest obstacles - " + ", ".join(context) if context else "No obstacles detected"

    def classify_scene(self):
        if self.latest_rgb is None:
            return
        self.classifying = True
        try:
            # Resize for API efficiency
            frame = cv2.resize(self.latest_rgb, (640, 360))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb_frame)
            
            # Encode to base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            img_b64 = base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
            
            depth_context = self.get_depth_context()

            prompt = f"""You are an autonomous vehicle perception system. Analyze this camera frame and respond in this exact format:

SCENE: [urban/highway/suburban/parking/intersection/tunnel/residential]
HAZARDS: [list any pedestrians, vehicles, obstacles, or none]
ROAD: [dry/wet/icy/construction/unknown]
TRAFFIC: [none/light/moderate/heavy/stopped]
VISIBILITY: [good/moderate/poor]
ACTION: [recommended driving action in one sentence]

Depth sensor data: {depth_context}
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
            self.get_logger().info(f"\n{'='*50}\n{self.last_result}\n{'='*50}")

        except Exception as e:
            self.get_logger().error(f"Classification error: {str(e)}")
            self.last_result = f"Error: {str(e)}"
        finally:
            self.classifying = False

    def display_frame(self):
        if self.latest_rgb is None:
            return
        display = self.latest_rgb.copy()
        
        # Dark overlay for text background
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (640, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        # Draw results
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

        # Status indicator
        status = "ANALYZING..." if self.classifying else "LIVE"
        color = (0, 165, 255) if self.classifying else (0, 255, 0)
        cv2.putText(display, status, (display.shape[1]-130, 25),
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

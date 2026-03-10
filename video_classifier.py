#!/usr/bin/env python3
import cv2
import numpy as np
import anthropic
import base64
import threading
import time
from PIL import Image as PILImage
from ultralytics import YOLO
import io
import sys

# ── Config ────────────────────────────────────────────────────────────
VIDEO_PATH = "/home/autobot/av_vla_project/driving_sample.mp4"
CLASSIFY_EVERY = 60  # Run Claude every 60 frames (~2 sec at 30fps)

class VideoSceneClassifier:
    def __init__(self):
        self.client = anthropic.Anthropic()

        print("Loading YOLOv8...")
        self.yolo = YOLO('yolov8n.pt')
        print("YOLOv8 ready!")

        self.last_result = "Initializing..."
        self.last_detections = []
        self.classifying = False
        self.frame_count = 0

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
                    'label': label, 'conf': conf,
                    'box': (x1, y1, x2, y2)
                })

    def get_yolo_summary(self):
        if not self.last_detections:
            return "No objects detected"
        counts = {}
        for d in self.last_detections:
            counts[d['label']] = counts.get(d['label'], 0) + 1
        return ", ".join([f"{v}x {k}" for k, v in counts.items()])

    def classify_scene(self, frame):
        self.classifying = True
        try:
            small = cv2.resize(frame, (640, 360))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            img_b64 = base64.standard_b64encode(
                buffer.getvalue()).decode('utf-8')

            yolo_context = self.get_yolo_summary()

            prompt = f"""You are an autonomous vehicle perception system. Analyze this dashcam frame and respond in this exact format:

SCENE: [urban/highway/suburban/parking/intersection/tunnel/residential]
HAZARDS: [list any immediate dangers or none]
ROAD: [dry/wet/icy/construction/unknown]
TRAFFIC: [none/light/moderate/heavy/stopped]
VISIBILITY: [good/moderate/poor]
ACTION: [recommended driving action in one sentence]

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
            print(f"\n{'='*50}\n{self.last_result}\n{'='*50}")

        except Exception as e:
            print(f"API Error: {e}")
            self.last_result = f"API Error: {str(e)[:60]}"
        finally:
            self.classifying = False

    def draw_boxes(self, frame):
        color_map = {
            'person':        (0, 0, 255),
            'car':           (0, 255, 0),
            'truck':         (0, 165, 255),
            'bus':           (0, 165, 255),
            'bicycle':       (255, 0, 0),
            'motorcycle':    (255, 0, 0),
            'traffic light': (0, 255, 255),
            'stop sign':     (0, 0, 255),
        }
        for det in self.last_detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            conf = det['conf']
            color = color_map.get(label, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-6),
                         (x1+tw+4, y1), color, -1)
            cv2.putText(frame, text, (x1+2, y1-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame

    def draw_overlay(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (720, 190), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 25
        for line in self.last_result.strip().split('\n'):
            if line.strip():
                color = (0, 255, 0)
                if 'HAZARD' in line and 'none' not in line.lower():
                    color = (0, 0, 255)
                elif 'ACTION' in line:
                    color = (0, 255, 255)
                cv2.putText(frame, line.strip(), (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y += 27

        # YOLO summary at bottom
        cv2.putText(frame, f"YOLO: {self.get_yolo_summary()}",
                   (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)

        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}",
                   (frame.shape[1]-150, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        # Status
        status = "ANALYZING..." if self.classifying else "LIVE"
        color = (0, 165, 255) if self.classifying else (0, 255, 0)
        cv2.putText(frame, status, (frame.shape[1]-140, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {fps:.0f}fps, {total} frames")
        print("Press Q to quit, SPACE to pause")

        paused = False
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Video ended. Replaying...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                self.frame_count += 1

                # YOLO on every frame
                self.run_yolo(frame)

                # Claude API every N frames
                if self.frame_count % CLASSIFY_EVERY == 0 and not self.classifying:
                    t = threading.Thread(
                        target=self.classify_scene, args=(frame.copy(),))
                    t.daemon = True
                    t.start()

                # Draw boxes and overlay
                frame = self.draw_boxes(frame)
                frame = self.draw_overlay(frame)

            cv2.imshow("AV Scene Classifier - Video Mode", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    classifier = VideoSceneClassifier()
    classifier.run(video)

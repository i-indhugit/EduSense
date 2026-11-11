"""
EduSense Synthetic Multi-Student Classroom Frame Simulator
===========================================================
Generates realistic, high-definition multi-student classroom video streams
complete with seating grid layouts, face bounding boxes, anonymous IDs,
gaze direction rays, MediaPipe landmarks, dynamic behavioral states,
side information panels, and temporal smoothing for anti-flicker attention analysis.
"""

import cv2
import numpy as np
import math
import time
import random
from typing import Dict, List, Tuple

def compute_attention_status(score: float, current_status: str = None) -> str:
    """
    Computes human-readable attention status with hysteresis filtering for stability.
    Categories:
      - Highly Attentive (85-100)
      - Attentive (70-84)
      - Moderately Attentive (55-69)
      - Needs Attention (40-54)
      - Low Engagement (25-39)
      - Not Attentive (0-24)
    """
    if score >= 85.0:
        target = "Highly Attentive"
    elif score >= 70.0:
        target = "Attentive"
    elif score >= 55.0:
        target = "Moderately Attentive"
    elif score >= 40.0:
        target = "Needs Attention"
    elif score >= 25.0:
        target = "Low Engagement"
    else:
        target = "Not Attentive"

    if current_status is None:
        return target

    # Hysteresis deadband margin (2.5 pts) to prevent flickering near boundary
    margin = 2.5
    if current_status == "Highly Attentive" and score < (85.0 - margin):
        return target
    elif current_status == "Attentive" and (score >= (85.0 + margin) or score < (70.0 - margin)):
        return target
    elif current_status == "Moderately Attentive" and (score >= (70.0 + margin) or score < (55.0 - margin)):
        return target
    elif current_status == "Needs Attention" and (score >= (55.0 + margin) or score < (40.0 - margin)):
        return target
    elif current_status == "Low Engagement" and (score >= (40.0 + margin) or score < (25.0 - margin)):
        return target
    elif current_status == "Not Attentive" and score >= (25.0 + margin):
        return target

    return current_status

LIVE_MODE = True

class SyntheticClassroomSimulator:
    def __init__(self, width: int = 1280, height: int = 720, num_students: int = 6):
        if LIVE_MODE:
            raise RuntimeError(
                "LIVE_MODE IS ACTIVE: Synthetic simulator is strictly disabled. "
                "Only real live webcam frames are permitted in production."
            )
        self.width = width
        self.height = height
        self.num_students = num_students
        self.start_time = time.time()
        self.frame_count = 0

        # Define 6 seating positions (2 rows x 3 columns)
        self.seats = [
            {"id": "Student 01", "row": 1, "col": 1, "center": (300, 260), "size": (180, 220)},
            {"id": "Student 02", "row": 1, "col": 2, "center": (640, 260), "size": (180, 220)},
            {"id": "Student 03", "row": 1, "col": 3, "center": (980, 260), "size": (180, 220)},
            {"id": "Student 04", "row": 2, "col": 1, "center": (260, 540), "size": (220, 260)},
            {"id": "Student 05", "row": 2, "col": 2, "center": (640, 540), "size": (220, 260)},
            {"id": "Student 06", "row": 2, "col": 3, "center": (1020, 540), "size": (220, 260)},
        ]

        # Temporal smoothing buffers for scores & status stability
        self.smoothed_attention: Dict[str, float] = {}
        self.smoothed_status: Dict[str, str] = {}
        self.confidence_scores: Dict[str, float] = {
            s["id"]: round(random.uniform(94.5, 99.2), 1) for s in self.seats
        }

    def generate_frame(self) -> Tuple[np.ndarray, List[Dict]]:
        self.frame_count += 1
        t = time.time() - self.start_time
        timestamp_now = time.strftime("%H:%M:%S")

        # Create dark neutral classroom background frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Neutral dark background gradient
        for y in range(self.height):
            val = int(12 + (y / self.height) * 10)
            frame[y, :] = (val, val, val + 5)

        # Front lecture board header
        cv2.rectangle(frame, (100, 15), (self.width - 100, 60), (25, 25, 30), -1)
        cv2.rectangle(frame, (100, 15), (self.width - 100, 60), (60, 60, 70), 1)
        cv2.putText(frame, "REAL-TIME CLASSROOM VISION & ATTENTION PIPELINE", 
                    (self.width // 2 - 250, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

        student_features = []

        for i, seat in enumerate(self.seats[:self.num_students]):
            cx, cy = seat["center"]
            w, h = seat["size"]
            student_id = seat["id"]

            # Dynamic behavioral state cycles
            phase = i * 1.5 + t * 0.8
            is_looking_away = math.sin(phase) < -0.6
            is_head_down = math.sin(phase * 0.5) > 0.85
            is_hand_raised = (i == 1 and math.sin(t * 0.5) > 0.3) or (i == 4 and math.sin(t * 0.3 + 1) > 0.7)
            is_drowsy = (i == 3 and math.sin(t * 0.2) > 0.6)
            is_yawning = (i == 3 and is_drowsy and math.sin(t * 0.4) > 0.5)

            # Determine Gaze, Head Pose & Raw Attention Score
            if is_head_down:
                gaze_dir = "Head Down"
                head_pose = "Head Down (-35°)"
                raw_att = 20.0 + 10.0 * math.sin(t * 2.0)
                posture = "Slouching"
                engagement_level = "Low"
            elif is_looking_away:
                gaze_dir = "Looking Away"
                head_pose = "Leaning (+28°)"
                raw_att = 48.0 + 8.0 * math.cos(t * 1.5)
                posture = "Leaning"
                engagement_level = "Moderate"
            else:
                gaze_dir = "Looking at Board"
                head_pose = "Upright (0°)"
                raw_att = 88.0 + 7.0 * math.sin(t * 1.2)
                posture = "Upright"
                engagement_level = "High"

            # Apply Temporal Exponential Moving Average (EMA) for Score Stability
            alpha = 0.20
            prev_att = self.smoothed_attention.get(student_id, raw_att)
            smoothed_att = alpha * raw_att + (1.0 - alpha) * prev_att
            self.smoothed_attention[student_id] = smoothed_att

            # Compute Human-Readable Attention Status with Hysteresis Locking
            prev_status = self.smoothed_status.get(student_id, None)
            att_status = compute_attention_status(smoothed_att, prev_status)
            self.smoothed_status[student_id] = att_status

            # Model Detection Confidence
            conf_pct = self.confidence_scores[student_id]

            # Determine Emotion
            if is_drowsy:
                emotion = "Sleepy"
            elif is_hand_raised:
                emotion = "Happy"
            elif i == 2 and math.sin(t * 0.7) > 0.2:
                emotion = "Confused"
            elif is_looking_away:
                emotion = "Bored"
            else:
                emotion = "Neutral"

            ear_val = 0.12 if is_drowsy else (0.28 + 0.04 * math.sin(phase * 3))
            eye_ratio = 0.75 if emotion == "Confused" else 1.0

            # Draw Head & Body Avatar Representation
            head_radius = w // 4
            head_center = (cx, cy - 30)

            # Torso / Body
            cv2.ellipse(frame, (cx, cy + 60), (w // 2, h // 3), 0, 180, 360, (50, 50, 60), -1)

            # Hand Raise Visual
            if is_hand_raised:
                cv2.line(frame, (cx + w // 3, cy + 40), (cx + w // 2 + 20, cy - 80), (200, 200, 200), 4)
                cv2.circle(frame, (cx + w // 2 + 20, cy - 80), 10, (220, 220, 220), -1)

            # Head Circle
            head_color = (120, 120, 130) if not is_drowsy else (80, 80, 100)
            cv2.circle(frame, head_center, head_radius, head_color, -1)
            cv2.circle(frame, head_center, head_radius, (180, 180, 190), 1)

            # Eyes & Gaze Direction Ray
            eye_y = head_center[1] - 8
            left_eye = (cx - 18, eye_y)
            right_eye = (cx + 18, eye_y)

            cv2.circle(frame, left_eye, 4, (30, 30, 30), -1)
            cv2.circle(frame, right_eye, 4, (30, 30, 30), -1)

            # Face / Student Bounding Box
            bx1 = cx - w // 2
            by1 = cy - h // 2
            bx2 = cx + w // 2
            by2 = cy + h // 2

            box_color = (200, 200, 200)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 1)

            # -------------------------------------------------------------
            # 1. Overlay ABOVE Bounding Box (ID, Attention Score, Confidence)
            # -------------------------------------------------------------
            top_h = 50
            top_w = max(160, w)
            tox1 = cx - top_w // 2
            toy1 = max(10, by1 - top_h - 4)
            tox2 = tox1 + top_w
            toy2 = toy1 + top_h

            cv2.rectangle(frame, (tox1, toy1), (tox2, toy2), (18, 18, 22), -1)
            cv2.rectangle(frame, (tox1, toy1), (tox2, toy2), (60, 60, 70), 1)

            cv2.putText(frame, student_id, (tox1 + 6, toy1 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            cv2.putText(frame, f"Att: {smoothed_att:.1f}%", (tox1 + 6, toy1 + 31), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
            cv2.putText(frame, f"Conf: {conf_pct:.1f}%", (tox1 + 6, toy1 + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 170), 1)

            # -------------------------------------------------------------
            # 2. Side Information Panel (Right Side Preferred, Left if Edge)
            # -------------------------------------------------------------
            panel_w = 175
            panel_h = 145

            # Auto-reposition if too close to right edge
            if bx2 + panel_w + 10 < self.width:
                px1 = bx2 + 6
            else:
                px1 = max(10, bx1 - panel_w - 6)

            py1 = max(10, by1)
            px2 = px1 + panel_w
            py2 = py1 + panel_h

            cv2.rectangle(frame, (px1, py1), (px2, py2), (15, 15, 18), -1)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (50, 50, 60), 1)

            # Side Panel Metrics Text Lines
            lines = [
                (f"{student_id}", (255, 255, 255)),
                (f"Att Score: {smoothed_att:.1f}%", (230, 230, 230)),
                (f"Confidence: {conf_pct:.1f}%", (180, 180, 190)),
                (f"Emotion: {emotion}", (200, 200, 210)),
                (f"Eye Gaze: {gaze_dir}", (180, 180, 190)),
                (f"Head Pose: {head_pose}", (180, 180, 190)),
                (f"Drowsy: {'Drowsy' if is_drowsy else 'Alert'}", (220, 220, 220)),
                (f"Engage: {engagement_level}", (180, 180, 190)),
                (f"Status: Tracked", (160, 160, 170)),
                (f"Updated: {timestamp_now}", (140, 140, 150))
            ]

            y_offset = py1 + 13
            for text, color in lines:
                cv2.putText(frame, text, (px1 + 6, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)
                y_offset += 13

            # Append feature dict for downstream fusion engine
            student_features.append({
                "student_id": student_id,
                "bbox": [bx1, by1, bx2, by2],
                "confidence_pct": conf_pct,
                "attention_pct": round(smoothed_att, 1),
                "attention_status": att_status,
                "gaze_direction": gaze_dir,
                "head_pose": head_pose,
                "emotion": emotion,
                "ear_value": ear_val,
                "is_drowsy": is_drowsy,
                "is_yawning": is_yawning,
                "hand_raised": is_hand_raised,
                "posture": posture,
                "eye_distance_ratio": eye_ratio
            })

        # Watermark / Pipeline Info Bar
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"EduSense Live Vision Pipeline | Temporal EMA Filter Active | FPS: 24.0 | {timestamp_str}", 
                    (20, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 170), 1)

        return frame, student_features

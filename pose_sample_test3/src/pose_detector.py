import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_pose(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        if results.pose_landmarks:
            self._draw_landmarks(image, results.pose_landmarks)
        
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    def _draw_landmarks(self, image, landmarks):
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

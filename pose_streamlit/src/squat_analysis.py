import mediapipe as mp
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class SquatStandards:
    """운동역학 및 스포츠 의학 연구 기반 스쿼트 기준"""
    STANDING = {
        'hip_angle': 172,
        'knee_angle': 175,
        'ankle_angle': 80,
        'tolerance': 5
    }
    
    PARALLEL = {
        'hip_angle': 95,
        'knee_angle': 90,
        'ankle_angle': 70,
        'tolerance': 8
    }
    
    DEPTH_LIMITS = {
        'min_hip': 50,
        'max_knee': 135,
        'ankle_range': (35, 45)
    }
    
    FORM_CHECKS = {
        'knee_tracking': {
            'description': 'Knees should track over toes',
            'tolerance': 12
        },
        'back_angle': {
            'min': 50,
            'max': 85
        },
        'weight_distribution': {
            'front': 0.45,
            'back': 0.55
        }
    }

class SquatGuide:
    """스쿼트 가이드라인 시각화 클래스"""
    def __init__(self):
        self.guide_points = []
        self.target_area = None
        self.calibration_complete = False

    def draw_guide_area(self, frame: np.ndarray, landmarks: List, target_area: Dict = None) -> np.ndarray:
        """가이드 영역 시각화"""
        return frame

class EnhancedSquatAnalysisEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        self.guide = SquatGuide()
        self.standards = SquatStandards()
        self.calibration_mode = True
        self.calibration_count = 0
        self.target_squat_area = None
        self.rep_count = 0
        self.current_phase = "STANDING"

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """프레임 분석 및 피드백 생성"""
        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 포즈 감지
        results = self.pose.process(image)
        
        # 이미지를 BGR로 다시 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            return image, {
                'is_correct': False,
                'feedback': 'No pose detected. Please stand in the frame.',
                'phase': 'UNKNOWN'
            }

        # 관절 포인트와 연결선 그리기
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        # 각도 계산 및 분석
        angles = self._calculate_joint_angles(results.pose_landmarks.landmark)
        
        # 캘리브레이션 모드 처리
        if self.calibration_mode:
            return self._handle_calibration(image, results.pose_landmarks)

        # 일반 분석 모드
        phase_feedback = self._analyze_squat_phase(angles, results.pose_landmarks)
        in_target_area = self._is_in_target_area(results.pose_landmarks.landmark)

        # 가이드라인 그리기
        image = self.guide.draw_guide_area(image, results.pose_landmarks.landmark, self.target_squat_area)

        # 각도 표시 (선택적)
        if hasattr(self, 'show_angles') and self.show_angles:
            self._draw_angles(image, angles)

        # 피드백 생성
        feedback = {
            'is_correct': in_target_area and phase_feedback['is_correct'],
            'feedback': phase_feedback['feedback'] if in_target_area else "Move to the guide area",
            'phase': self.current_phase,
            'angles': angles,
            'score': phase_feedback.get('score', 0)
        }

        return image, feedback

    def _draw_angles(self, image: np.ndarray, angles: Dict[str, float]) -> None:
        """관절 각도 시각화"""
        height, width = image.shape[:2]
        text_color = (255, 255, 255)  # 흰색
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # 각도 텍스트 위치 설정
        positions = {
            'hip': (int(width * 0.1), int(height * 0.5)),
            'knee': (int(width * 0.1), int(height * 0.6))
        }

        for joint, angle in angles.items():
            if joint in positions:
                text = f"{joint.title()}: {angle:.1f}°"
                cv2.putText(image, text, positions[joint], font, font_scale, text_color, thickness)
    
    def _calculate_joint_angles(self, landmarks) -> Dict[str, float]:
        """주요 관절 각도 계산"""
        def calculate_angle(a, b, c) -> float:
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                     np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
            return angle

        # 주요 관절 포인트 추출
        hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        # 각도 계산
        hip_angle = calculate_angle(shoulder, hip, knee)
        knee_angle = calculate_angle(hip, knee, ankle)
        
        return {
            'hip': hip_angle,
            'knee': knee_angle
        }

    def _is_valid_squat_position(self, landmarks) -> bool:
        """올바른 스쿼트 자세인지 확인"""
        angles = self._calculate_joint_angles(landmarks)

        # 병렬 스쿼트 기준에 맞는지 확인
        hip_in_range = abs(angles['hip'] - self.standards.PARALLEL['hip_angle']) <= self.standards.PARALLEL['tolerance']
        knee_in_range = abs(angles['knee'] - self.standards.PARALLEL['knee_angle']) <= self.standards.PARALLEL['tolerance']

        return hip_in_range and knee_in_range

    def _handle_calibration(self, image: np.ndarray, landmarks) -> Tuple[np.ndarray, Dict]:
        """캘리브레이션 처리"""
        if self.calibration_count >= 5:
            self.calibration_mode = False
            self.target_squat_area = self._calculate_target_area(landmarks.landmark)
            return image, {
                'is_correct': True,
                'feedback': 'Calibration complete! Ready for workout.',
                'phase': 'CALIBRATED'
            }

        if self._is_valid_squat_position(landmarks.landmark):
            self.calibration_count += 1
            feedback = f'Calibration rep {self.calibration_count}/5 recorded'
        else:
            feedback = 'Perform a proper squat for calibration'

        return image, {
            'is_correct': False,
            'feedback': feedback,
            'phase': 'CALIBRATING'
        }

    def _calculate_target_area(self, landmarks) -> Dict:
        """캘리브레이션 시 가상의 의자 모양을 정의"""
        self.guide.target_area = {
            'hip': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            'knee': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            'ankle': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        }
        return self.guide.target_area

    def _is_in_target_area(self, landmarks) -> bool:
        """현재 자세가 가상의 의자 모양과 일치하는지 확인"""
        if not self.guide.target_area:
            return False

        current_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        target_hip = self.guide.target_area['hip']

        tolerance = 0.05  # 위치 허용 오차
        hip_diff = np.hypot(current_hip.x - target_hip.x, current_hip.y - target_hip.y)

        return hip_diff < tolerance

    def _analyze_squat_phase(self, angles: Dict[str, float], landmarks) -> Dict:
        """스쿼트 단계 분석 및 피드백 생성"""
        result = {
            'is_correct': False,
            'feedback': '',
            'score': 0
        }

        # 현재 엉덩이 높이 계산
        hip_height = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].y

        # 단계 판단
        if angles['hip'] > 160:  # 서있는 자세
            self.current_phase = "STANDING"
            if not hasattr(self, 'prev_hip_height'):
                self.prev_hip_height = hip_height
        elif angles['hip'] < 100:  # 스쿼트 자세
            self.current_phase = "SQUAT"

        result['phase'] = self.current_phase

        # 자세 분석
        if self.current_phase == "SQUAT":
            hip_deviation = abs(angles['hip'] - self.standards.PARALLEL['hip_angle'])
            knee_deviation = abs(angles['knee'] - self.standards.PARALLEL['knee_angle'])

            angle_score = max(0, 100 - (hip_deviation + knee_deviation))
            result['score'] = angle_score

            if angle_score >= 90:
                result['is_correct'] = True
                result['feedback'] = "Good!"
                if not hasattr(self, 'rep_counted') or not self.rep_counted:
                    self.rep_count += 1
                    self.rep_counted = True
            else:
                result['feedback'] = "Try again!"

        elif self.current_phase == "STANDING":
            self.rep_counted = False
            result['feedback'] = "Ready"
            result['is_correct'] = True

        return result

    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'pose'):
            self.pose.close()
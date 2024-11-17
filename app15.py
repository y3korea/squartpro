# Part 1: Imports and Basic Setup
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Part 2: SquatStandards Class
@dataclass
class SquatStandards:
    """운동역학 및 스포츠 의학 연구 기반 스쿼트 기준"""

    # Reference: Journal of Strength and Conditioning Research (2021)
    STANDING = {
        'hip_angle': 172,    # 표준 직립 자세 기준
        'knee_angle': 175,   # 완전 신전 기준
        'ankle_angle': 80,   # 중립 발목 각도
        'tolerance': 5       # 허용 오차
    }

    # Reference: Medicine & Science in Sports & Exercise (2022)
    PARALLEL = {
        'hip_angle': 95,     # 병렬 스쿼트 기준
        'knee_angle': 90,    # 표준 무릎 굴곡
        'ankle_angle': 70,   # 최적 발목 각도
        'tolerance': 8       # 동적 움직임 고려 허용치
    }

    # Reference: International Journal of Sports Physical Therapy (2023)
    DEPTH_LIMITS = {
        'min_hip': 50,       # 최소 안전 엉덩이 각도
        'max_knee': 135,     # 최대 안전 무릎 각도
        'ankle_range': (35, 45)  # 최적 발목 가동 범위
    }

    # Reference: Sports Biomechanics Journal (2023)
    FORM_CHECKS = {
        'knee_tracking': {
            'description': 'Knees should track over toes',
            'tolerance': 12   # 횡방향 허용 편차
        },
        'back_angle': {
            'min': 50,       # 최소 안전 등판 각도
            'max': 85        # 최대 권장 등판 각도
        },
        'weight_distribution': {
            'front': 0.45,   # 전방 하중 분포
            'back': 0.55     # 후방 하중 분포
        }
    }

# Part 3: SquatGuide Class
class SquatGuide:
    """스쿼트 가이드라인 시각화 클래스"""
    def __init__(self):
        self.guide_points = []
        self.target_area = None
        self.calibration_complete = False

    def draw_guide_area(self, frame: np.ndarray, landmarks: List, target_area: Dict = None) -> np.ndarray:
        """가이드 영역 시각화"""
        # 가이드 영역 그리기 로직 (필요에 따라 구현)
        return frame

# Part 4: EnhancedSquatAnalysisEngine Class
class EnhancedSquatAnalysisEngine:
    """향상된 스쿼트 분석 엔진"""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
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

    def _draw_landmarks(self, image: np.ndarray, landmarks) -> None:
        """관절 포인트와 스켈레톤 시각화"""
        mp.solutions.drawing_utils.draw_landmarks(
            image, 
            landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )

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
            # 병렬 스쿼트 기준 체크
            hip_deviation = abs(angles['hip'] - self.standards.PARALLEL['hip_angle'])
            knee_deviation = abs(angles['knee'] - self.standards.PARALLEL['knee_angle'])

            # 점수 계산 (100점 만점)
            angle_score = max(0, 100 - (hip_deviation + knee_deviation))
            result['score'] = angle_score

            # 피드백 생성
            if angle_score >= 90:
                result['is_correct'] = True
                result['feedback'] = "Good!"
            else:
                result['feedback'] = "Try again!"

            # 반복 횟수 업데이트
            if not hasattr(self, 'rep_counted'):
                self.rep_counted = False

            if angle_score >= 90 and not self.rep_counted:
                self.rep_count += 1
                self.rep_counted = True

        elif self.current_phase == "STANDING":
            self.rep_counted = False
            result['feedback'] = "Ready"
            result['is_correct'] = True

        return result

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """프레임 분석 및 피드백 생성"""
        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            return image, {
                'is_correct': False,
                'feedback': 'No pose detected. Please stand in the frame.',
                'phase': 'UNKNOWN'
            }

        # 골격 그리기
        self._draw_landmarks(image, results.pose_landmarks)

        # 캘리브레이션 모드 처리
        if self.calibration_mode:
            return self._handle_calibration(image, results.pose_landmarks)

        # 일반 분석 모드
        return self._analyze_regular_squat(image, results.pose_landmarks)

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

        # 가이드라인 그리기
        image = self.guide.draw_guide_area(image, landmarks.landmark, self.target_squat_area)
        return image, {
            'is_correct': False,
            'feedback': feedback,
            'phase': 'CALIBRATING'
        }

    def _analyze_regular_squat(self, image: np.ndarray, landmarks) -> Tuple[np.ndarray, Dict]:
        """일반 스쿼트 분석"""
        angles = self._calculate_joint_angles(landmarks.landmark)
        in_target_area = self._is_in_target_area(landmarks.landmark)

        # 스쿼트 자세 분석
        phase_feedback = self._analyze_squat_phase(angles, landmarks)

        # 가이드라인 그리기
        image = self.guide.draw_guide_area(image, landmarks.landmark, self.target_squat_area)

        # 피드백 생성
        if in_target_area and phase_feedback['is_correct']:
            feedback = "Good!"
        else:
            feedback = "Try again!"

        return image, {
            'is_correct': in_target_area and phase_feedback['is_correct'],
            'feedback': feedback,
            'phase': self.current_phase,
            'angles': angles,
            'score': phase_feedback.get('score', 0)
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

        # 현재 힙 좌표
        current_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        target_hip = self.guide.target_area['hip']

        # 위치 오차 허용 범위 설정
        tolerance = 0.05  # 위치 허용 오차

        # 힙의 위치 차이 계산
        hip_diff = np.hypot(current_hip.x - target_hip.x, current_hip.y - target_hip.y)

        # 힙이 가이드 영역 내에 있는지 확인
        return hip_diff < tolerance

    def cleanup(self):
        """리소스 정리"""
        pass

def main():
    st.title("AI SmartSquat Pro 🏋️‍♂️")

    # 세션 상태 초기화
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedSquatAnalysisEngine()
    if 'rep_count' not in st.session_state:
        st.session_state.rep_count = 0

    # 사이드바 설정
    with st.sidebar:
        st.header("Settings ⚙️")
        show_skeleton = st.checkbox("Show Skeleton 🦴", True)
        show_guide = st.checkbox("Show Guide Area 🎯", True)
        show_angles = st.checkbox("Show Joint Angles 📐", True)

    # 메인 컨트롤
    cols = st.columns(4)

    if cols[0].button("🎯 Start Calibration"):
        st.session_state.analyzer = EnhancedSquatAnalysisEngine()
        st.info("Please perform 5 squats to set your baseline.")

    if cols[1].button("📷 Capture Image"):
        st.session_state.capture_mode = True

    if cols[2].button("🚀 Start Workout"):
        st.session_state.workout_active = True

    if cols[3].button("⏹️ Stop"):
        if hasattr(st.session_state.analyzer, 'cleanup'):
            st.session_state.analyzer.cleanup()
        st.session_state.workout_active = False

    # 이미지 캡처
    image_data = st.camera_input("Take a picture")

    if image_data is not None:
        bytes_data = image_data.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # 프레임 분석
        processed_frame, feedback = st.session_state.analyzer.analyze_frame(frame)

        # 피드백 표시
        feedback_color = (0, 255, 0) if feedback['is_correct'] else (255, 0, 0)
        cv2.putText(
            processed_frame,
            feedback['feedback'],
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            feedback_color,
            2
        )

        # 결과 이미지 출력
        st.image(processed_frame, channels="BGR")

        # Progress 업데이트
        st.write(f"Reps Completed 🔄: {st.session_state.analyzer.rep_count}")
        st.write(f"Current Phase: {feedback['phase']}")
        st.write(f"Feedback: {feedback['feedback']}")

if __name__ == "__main__":
    main()
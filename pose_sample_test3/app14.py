# Part 1: Imports and Basic Setup
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pyttsx3
import threading

class VoiceFeedback:
    """실시간 음성 피드백 시스템"""
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            # 사용 가능한 음성이 있다면 첫 번째 음성 선택
            if voices:
                self.engine.setProperty('voice', voices[0].id)
            self.engine.setProperty('rate', 150)    # 속도 설정
            self.engine.setProperty('volume', 0.9)  # 볼륨 설정
        except Exception as e:
            print(f"Failed to initialize voice engine: {str(e)}")
            self.engine = None
        
        self.speaking_thread = None
        self.last_feedback = ""
        
    def speak(self, text: str):
        """비동기 음성 출력"""
        if not self.engine:
            return
            
        if text == self.last_feedback:  # 같은 피드백 반복 방지
            return
            
        self.last_feedback = text
        
        def speak_worker():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Voice feedback error: {str(e)}")
        
        try:
            # 이전 음성이 아직 재생 중이면 중단
            if self.speaking_thread and self.speaking_thread.is_alive():
                if hasattr(self.engine, 'stop'):
                    self.engine.stop()
                self.speaking_thread.join(timeout=1.0)
            
            # 새로운 음성 시작
            self.speaking_thread = threading.Thread(target=speak_worker)
            self.speaking_thread.start()
        except Exception as e:
            print(f"Thread error: {str(e)}")

    def cleanup(self):
        """음성 엔진 정리"""
        try:
            if self.speaking_thread and self.speaking_thread.is_alive():
                if hasattr(self.engine, 'stop'):
                    self.engine.stop()
                self.speaking_thread.join(timeout=1.0)
            if self.engine:
                self.engine.stop()
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="AI SmartSquat Pro",
    page_icon="🤺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .main { padding: 0rem 0rem; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .guide-area {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .feedback-text {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .success-feedback {
        color: #4CAF50;
    }
    .warning-feedback {
        color: #ff9800;
    }
</style>
""", unsafe_allow_html=True)

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
        height, width = frame.shape[:2]
        overlay = frame.copy()

        if target_area:
            # 가상의 의자 모양 그리기
            # 캘리브레이션 시 저장한 힙, 무릎, 발목 좌표 사용
            hip = target_area['hip']
            knee = target_area['knee']
            ankle = target_area['ankle']

            # 좌표 변환
            hip_point = (int(hip.x * width), int(hip.y * height))
            knee_point = (int(knee.x * width), int(knee.y * height))
            ankle_point = (int(ankle.x * width), int(ankle.y * height))

            # 의자 좌석 부분 생성 (엉덩이 위치)
            seat_width = abs(hip_point[0] - knee_point[0]) * 1.2
            seat_height = abs(hip_point[1] - knee_point[1]) * 0.5

            # 의자 좌석의 좌표 계산
            seat_top_left = (int(hip_point[0] - seat_width / 2), int(hip_point[1]))
            seat_top_right = (int(hip_point[0] + seat_width / 2), int(hip_point[1]))
            seat_bottom_left = (int(hip_point[0] - seat_width / 2), int(hip_point[1] + seat_height))
            seat_bottom_right = (int(hip_point[0] + seat_width / 2), int(hip_point[1] + seat_height))

            # 등받이 부분 생성
            backrest_height = seat_height * 1.5
            backrest_top_left = (seat_top_left[0], int(seat_top_left[1] - backrest_height))
            backrest_top_right = (seat_top_right[0], int(seat_top_right[1] - backrest_height))

            # 의자 모양을 위한 폴리곤 포인트 생성
            chair_points = np.array([
                backrest_top_left,
                backrest_top_right,
                seat_top_right,
                seat_bottom_right,
                seat_bottom_left,
                seat_top_left
            ], np.int32)

            # 가이드라인 그리기
            cv2.polylines(
                overlay,
                [chair_points],
                True,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # 가이드 영역 채우기
            cv2.fillPoly(overlay, [chair_points], (0, 255, 0, 64))

            # 오버레이 블렌딩
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        return frame

# Part 4: EnhancedSquatAnalysisEngine Class
class EnhancedSquatAnalysisEngine:
    """향상된 스쿼트 분석 엔진"""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.guide = SquatGuide()
        self.standards = SquatStandards()
        self.calibration_mode = True
        self.calibration_count = 0
        self.target_squat_area = None
        self.rep_count = 0
        self.current_phase = "STANDING"
        self.voice = VoiceFeedback()
        
        # 데이터 저장 경로 설정 및 생성
        self.data_path = os.path.join(os.path.expanduser('~'), 'squat_data')
        for folder in ['daily', 'weekly', 'monthly']:
            path = os.path.join(self.data_path, folder)
            os.makedirs(path, exist_ok=True)

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

    def _draw_landmarks(self, image: np.ndarray, landmarks: List) -> None:
        """관절 포인트와 스켈레톤 시각화"""
        height, width = image.shape[:2]
        
        # 스켈레톤 연결선 그리기
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = (int(landmarks[start_idx].x * width), 
                         int(landmarks[start_idx].y * height))
            end_point = (int(landmarks[end_idx].x * width), 
                        int(landmarks[end_idx].y * height))
            
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        # 관절 포인트 그리기
        for idx, landmark in enumerate(landmarks):
            pos = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(image, pos, 5, (0, 0, 255), -1)
            
            # 주요 관절 각도 표시
            if idx in [self.mp_pose.PoseLandmark.LEFT_HIP.value,
                      self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                      self.mp_pose.PoseLandmark.LEFT_ANKLE.value]:
                angles = self._calculate_joint_angles(landmarks)
                if idx == self.mp_pose.PoseLandmark.LEFT_HIP.value:
                    angle_text = f"Hip: {angles['hip']:.1f}°"
                elif idx == self.mp_pose.PoseLandmark.LEFT_KNEE.value:
                    angle_text = f"Knee: {angles['knee']:.1f}°"
                else:
                    continue
                    
                cv2.putText(image, angle_text,
                           (pos[0] + 10, pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)

    def _analyze_squat_phase(self, angles: Dict[str, float], landmarks: List) -> Dict:
        """스쿼트 단계 분석 및 피드백 생성"""
        result = {
            'is_correct': False,
            'feedback': '',
            'score': 0
        }

        # 현재 엉덩이 높이 계산
        hip_height = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y

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
        self._draw_landmarks(image, results.pose_landmarks.landmark)

        # 캘리브레이션 모드 처리
        if self.calibration_mode:
            return self._handle_calibration(image, results.pose_landmarks.landmark)

        # 일반 분석 모드
        return self._analyze_regular_squat(image, results.pose_landmarks.landmark)

    def _handle_calibration(self, image: np.ndarray, landmarks: List) -> Tuple[np.ndarray, Dict]:
        """캘리브레이션 처리"""
        if self.calibration_count >= 5:
            self.calibration_mode = False
            self.target_squat_area = self._calculate_target_area(landmarks)
            self.voice.speak("Calibration complete")
            return image, {
                'is_correct': True,
                'feedback': 'Calibration complete! Ready for workout.',
                'phase': 'CALIBRATED'
            }

        if self._is_valid_squat_position(landmarks):
            self.calibration_count += 1
            self.voice.speak(f"Good form {self.calibration_count}")
            feedback = f'Calibration rep {self.calibration_count}/5 recorded'
        else:
            feedback = 'Perform a proper squat for calibration'

        # 가이드라인 그리기
        image = self.guide.draw_guide_area(image, landmarks, self.target_squat_area)
        return image, {
            'is_correct': False,
            'feedback': feedback,
            'phase': 'CALIBRATING'
        }

    def _analyze_regular_squat(self, image: np.ndarray, landmarks: List) -> Tuple[np.ndarray, Dict]:
        """일반 스쿼트 분석"""
        angles = self._calculate_joint_angles(landmarks)
        in_target_area = self._is_in_target_area(landmarks)
        
        # 스쿼트 자세 분석
        phase_feedback = self._analyze_squat_phase(angles, landmarks)
        
        # 가이드라인 그리기
        image = self.guide.draw_guide_area(image, landmarks, self.target_squat_area)

        # 피드백 생성
        if in_target_area and phase_feedback['is_correct']:
            self.voice.speak("Good!")
            feedback = "Good!"
        else:
            self.voice.speak("Try again!")
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
        self.voice.cleanup()

def main():
    st.title("AI SmartSquat Pro 🏋️‍♂️")
    
    # 세션 상태 초기화
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedSquatAnalysisEngine()
    
    # 사이드바 설정
    with st.sidebar:
        st.header("Settings ⚙️")
        show_skeleton = st.checkbox("Show Skeleton 🦴", True)
        show_guide = st.checkbox("Show Guide Area 🎯", True)
        show_angles = st.checkbox("Show Joint Angles 📐", True)

    # 메인 컨트롤
    cols = st.columns(4)
    
    if cols[0].button("🎯 Start Calibration", use_container_width=True):
        st.session_state.analyzer = EnhancedSquatAnalysisEngine()
        st.info("Please perform 5 squats to set your baseline.")
    
    if cols[1].button("📹 Camera Check", use_container_width=True):
        st.session_state.camera_check = True
    
    if cols[2].button("🚀 Start Workout", use_container_width=True):
        st.session_state.workout_active = True
    
    if cols[3].button("⏹️ Stop", use_container_width=True):
        if hasattr(st.session_state.analyzer, 'cleanup'):
            st.session_state.analyzer.cleanup()
        st.session_state.workout_active = False

    # 카메라 피드
    FRAME_WINDOW = st.empty()
    
    # Progress 표시
    progress_cols = st.columns(2)
    rep_count = progress_cols[0].empty()
    phase_indicator = progress_cols[1].empty()

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video frame")
                break

            # 프레임 분석
            processed_frame, feedback = st.session_state.analyzer.analyze_frame(frame)
            
            # 피드백 표시
            feedback_color = (0, 255, 0) if feedback['is_correct'] else (0, 0, 255)
            cv2.putText(
                processed_frame,
                feedback['feedback'],
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                feedback_color,
                2
            )

            # Progress 업데이트
            rep_count.metric("Reps Completed 🔄", st.session_state.analyzer.rep_count)
            phase_indicator.info(f"Current Phase: {feedback['phase']}")

            # 스켈레톤과 가이드 표시 설정 적용
            if not show_skeleton:
                # 스켈레톤 숨기기 (이미지에서 제거하는 로직 필요)
                pass
            if not show_guide:
                # 가이드라인 숨기기 (이미지에서 제거하는 로직 필요)
                pass

            FRAME_WINDOW.image(processed_frame, channels="BGR")

            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        if hasattr(st.session_state.analyzer, 'cleanup'):
            st.session_state.analyzer.cleanup()

if __name__ == "__main__":
    main()

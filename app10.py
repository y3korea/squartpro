import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import openai  # 추가된 기능: OpenAI API 사용
import requests
import csv

# 페이지 설정은 첫 번째 호출로만 제한되도록 수정
st.set_page_config(
    page_title="AI SmartSquat Pro",
    page_icon="🤺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 앱 스타일 설정
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
    }
    .css-1d391kg {
        padding: 1rem 1rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .error-metric {
        color: #ff0000;
    }
    .success-metric {
        color: #00ff00;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'template_saved' not in st.session_state:
    st.session_state.template_saved = False
if 'current_score' not in st.session_state:
    st.session_state.current_score = 0
if 'rep_count' not in st.session_state:
    st.session_state.rep_count = 0
if 'user_history' not in st.session_state:
    st.session_state.user_history = []  # 추가된 기능: 유저의 과거 운동 기록 보관

# 데이터 저장 경로 설정
DATA_FOLDER = '/Users/y3korea/Documents/pose_sample_test2/squat_data'
WEEKLY_FOLDER = os.path.join(DATA_FOLDER, "weekly")
MONTHLY_FOLDER = os.path.join(DATA_FOLDER, "monthly")

# 필요한 폴더가 없으면 생성
for folder in [DATA_FOLDER, WEEKLY_FOLDER, MONTHLY_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# OpenAI API 키 입력 UI 추가
api_key_input = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# OpenAI API 설정 함수
def set_openai_api_key(api_key):
    openai.api_key = api_key

# 사용자가 입력한 API 키를 설정
if api_key_input:
    set_openai_api_key(api_key_input)

@dataclass
class SquatStandards:
    """의학적 연구 기반 스쿼트 기준"""
    STANDING = {
        'hip_angle': 170,    # NSCA 기준
        'knee_angle': 175,   # ACSM 기준
        'ankle_angle': 80,   # Sports Medicine Journal
        'tolerance': 5
    }
    
    PARALLEL = {
        'hip_angle': 95,     # NSCA 표준 병렬 스쿼트 각도
        'knee_angle': 90,    # ACSM 권장 무릎 각도
        'ankle_angle': 70,   # 스포츠 의학 연구 권장
        'tolerance': 10
    }
    
    DEPTH_LIMITS = {
        'min_hip': 45,       # 최소 안전 엉덩이 각도
        'max_knee': 140,     # 최대 안전 무릎 각도
        'ankle_range': (30, 45)  # 최적 발목 가동 범위
    }
    
    FORM_CHECKS = {
        'knee_tracking': {
            'description': 'Knees should track over toes',
            'tolerance': 15   # Maximum deviation in degrees
        },
        'back_angle': {
            'min': 45,       # Minimum safe back angle
            'max': 90        # Maximum effective back angle
        },
        'weight_distribution': {
            'front': 0.4,    # 40% front
            'back': 0.6      # 60% back (heel)
        }
    }

class SquatAnalysisEngine:
    """스쿼트 동작 분석 엔진"""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        self.standards = SquatStandards()
        self.current_phase = "STANDING"
        self.prev_hip_height = None
        self.rep_count = 0
        self.rep_counted = False
        self.start_time = None
        self.rep_speeds = []
        self.calibration_mode = True  # 초기 5회 스쿼트 기준 설정을 위해 추가된 변수
        self.calibration_count = 0
        self.target_squat_area = None
        # 추가된 기능: AI 기반 피드백 활성화
        self.use_ai_feedback = True
        self.api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'claude': os.getenv('CLAUDE_API_KEY'),
            'perplexity': os.getenv('PERPLEXITY_API_KEY'),
            'gemini': os.getenv('GEMINI_API_KEY')
        }
    
    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """프레임 분석 및 피드백 생성"""
        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        feedback_data = {
            'is_correct': False,
            'phase': self.current_phase,
            'angles': {},
            'feedback': '',
            'score': 0,
            'rep_count': self.rep_count
        }

        if results.pose_landmarks:
            self._draw_landmarks(image, results.pose_landmarks.landmark)
            if self.calibration_mode and self.calibration_count < 5:
                # 초기 기준 설정 모드
                if self._is_in_target_area(results.pose_landmarks.landmark):
                    self.calibration_count += 1
                    st.info(f"Calibration rep {self.calibration_count}/5 complete!")
                    
                    # 5회 기준 완료 시 가상 기준선 설정
                    if self.calibration_count == 5:
                        self.target_squat_area = self._calculate_target_area(results.pose_landmarks.landmark)
                        st.success("Calibration complete! You can now start your squat workout.")
                        self.calibration_mode = False
                else:
                    st.warning("Perform your squats to set the initial baseline.")
            else:
                # 기준 설정 후 스쿼트 평가
                angles = self._calculate_joint_angles(results.pose_landmarks.landmark)
                feedback_data['angles'] = angles
                
                if self._is_in_target_area(results.pose_landmarks.landmark):
                    feedback_data['rep_count'] = self.rep_count + 1
                    st.success("Good Job!")
                else:
                    st.error("Try Again!")

                # 자세 분석 및 피드백
                phase_feedback = self._analyze_squat_phase(
                    angles,
                    results.pose_landmarks.landmark
                )
                feedback_data.update(phase_feedback)

                # 랜드마크를 연결하는 스켈레톤 그리기
                self._draw_landmarks(image, results.pose_landmarks.landmark)

                # AI 피드백 생성 (추가된 기능)
                if self.use_ai_feedback:
                    ai_feedback = self.generate_ai_feedback(angles)
                    feedback_data['feedback'] += f" {ai_feedback}"

                # 횟수 카운팅
                self._update_rep_count(angles)

                # 좌표 데이터를 CSV 파일에 저장
                self._save_coordinates_to_csv(results.pose_landmarks.landmark)
        else:
            st.warning("Not all landmarks detected properly. Please ensure you are in the frame clearly and prepare for a squat.")

        return image, feedback_data

    def _calculate_target_area(self, landmarks) -> Dict:
        """엉덩이와 허벅지를 감싸는 가상의 영역을 계산합니다."""
        hip_point = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        knee_point = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]

        target_area = {
            'top_left': (hip_point.x - 0.05, hip_point.y - 0.05),
            'bottom_right': (knee_point.x + 0.05, knee_point.y + 0.05)
        }
        return target_area

    def _is_in_target_area(self, landmarks) -> bool:
        """사용자가 가상의 기준 영역 안에 있는지 확인합니다."""
        if not self.target_squat_area:
            return False

        hip_point = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        knee_point = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]

        in_area = (
            self.target_squat_area['top_left'][0] <= hip_point.x <= self.target_squat_area['bottom_right'][0] and
            self.target_squat_area['top_left'][1] <= hip_point.y <= self.target_squat_area['bottom_right'][1] and
            self.target_squat_area['top_left'][0] <= knee_point.x <= self.target_squat_area['bottom_right'][0] and
            self.target_squat_area['top_left'][1] <= knee_point.y <= self.target_squat_area['bottom_right'][1]
        )
        
        return in_area

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

        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
        ]
        for idx in required_landmarks:
            if idx >= len(landmarks) or landmarks[idx] is None:
                st.warning("Not all landmarks detected properly. Please ensure you are in the frame clearly and prepare for a squat.")
                return {
                    'hip': 0.0,
                    'knee': 0.0,
                    'ankle': 0.0
                }

        # 엉덩이 각도
        hip_angle = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        )

        # 무릎 각도
        knee_angle = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )

        # 발목 각도
        ankle_angle = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        )

        return {
            'hip': hip_angle,
            'knee': knee_angle,
            'ankle': ankle_angle
        }

    def _save_coordinates_to_csv(self, landmarks) -> None:
        """좌표 데이터를 CSV 파일에 저장"""
        current_time = datetime.now()
        csv_file_path = os.path.join(DATA_FOLDER, f"landmarks_{current_time.strftime('%Y-%m-%d')}.csv")
        weekly_csv_file_path = os.path.join(WEEKLY_FOLDER, f"weekly_{current_time.strftime('%Y-%U')}.csv")
        monthly_csv_file_path = os.path.join(MONTHLY_FOLDER, f"monthly_{current_time.strftime('%Y-%m')}.csv")
        
        fieldnames = ['timestamp', 'landmark', 'x', 'y', 'z']

        # 파일이 없으면 헤더 생성
        for path in [csv_file_path, weekly_csv_file_path, monthly_csv_file_path]:
            if not os.path.isfile(path):
                with open(path, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()

        # 랜드마크 데이터 기록
        for path in [csv_file_path, weekly_csv_file_path, monthly_csv_file_path]:
            with open(path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                for idx, landmark in enumerate(landmarks):
                    writer.writerow({
                        'timestamp': current_time.isoformat(),
                        'landmark': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })

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
            self.prev_hip_height = hip_height
        elif angles['hip'] < 100:  # 스쿼트 자세
            self.current_phase = "SQUAT"
        
        result['phase'] = self.current_phase

        # 자세 분석
        if self.current_phase == "SQUAT":
            # 병렬 스쿼트 기준 체크
            hip_deviation = abs(angles['hip'] - self.standards.PARALLEL['hip_angle'])
            knee_deviation = abs(angles['knee'] - self.standards.PARALLEL['knee_angle'])
            ankle_deviation = abs(angles['ankle'] - self.standards.PARALLEL['ankle_angle'])

            # 점수 계산 (100점 만점)
            angle_score = max(0, 100 - (hip_deviation + knee_deviation + ankle_deviation))
            result['score'] = angle_score

            # 피드백 생성
            if angle_score >= 90:
                result['is_correct'] = True
                result['feedback'] = "Perfect form! Hold this position."
            elif angle_score >= 70:
                result['feedback'] = "Good! Watch your depth and form."
            else:
                if hip_deviation > self.standards.PARALLEL['tolerance']:
                    result['feedback'] += "Adjust hip angle. "
                if knee_deviation > self.standards.PARALLEL['tolerance']:
                    result['feedback'] += "Check knee position. "
                if ankle_deviation > self.standards.PARALLEL['tolerance']:
                    result['feedback'] += "Mind your ankle mobility. "

        return result

    def _draw_landmarks(self, image: np.ndarray, landmarks) -> None:
        """랜드마크를 그리고 연결하는 스켈레톤을 화면에 표시"""
        height, width, _ = image.shape
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = (int(landmarks[start_idx].x * width), int(landmarks[start_idx].y * height))
            end_point = (int(landmarks[end_idx].x * width), int(landmarks[end_idx].y * height))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        for landmark in landmarks:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(image, point, 5, (0, 0, 255), -1)

    def _update_rep_count(self, angles: Dict[str, float]) -> None:
        """스쿼트 반복 횟수 업데이트"""
        # 스쿼트 완료 조건
        if (self.current_phase == "SQUAT" and
            angles['hip'] < 100 and
            angles['knee'] < 95):
            
            if not self.rep_counted:
                self.rep_count += 1
                self.rep_counted = True
                # 추가된 기능: 반복 속도 계산 및 기록
                if self.start_time is not None:
                    end_time = datetime.now()
                    rep_speed = (end_time - self.start_time).total_seconds()
                    self.rep_speeds.append(rep_speed)
                    self.start_time = end_time
                else:
                    self.start_time = datetime.now()
                
        # 초기 자세로 돌아왔을 때 카운트 가능 상태로 리셋
        elif angles['hip'] > 160:
            self.rep_counted = False

    def generate_ai_feedback(self, angles: Dict[str, float]) -> str:
        """AI 기반 피드백 생성"""
        prompt = (
            f"The user's hip angle is {angles['hip']} degrees, knee angle is {angles['knee']} degrees, "
            f"and ankle angle is {angles['ankle']} degrees. Provide feedback on the squat form."
        )
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=50
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return "AI feedback currently unavailable."

    def reset(self):
        """분석기 리셋"""
        self.rep_count = 0
        self.rep_counted = False
        self.current_phase = "STANDING"
        self.prev_hip_height = None
        self.start_time = None
        self.rep_speeds = []

# UI 버튼 추가 및 설정
st.title("AI SmartSquat Pro")

# 설정 컨트롤
with st.expander("Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        target_reps = st.number_input(
            "Target Reps",
            min_value=1,
            max_value=20,
            value=5
        )
        detection_confidence = st.slider(
            "Detection Confidence",
            0.0, 1.0, 0.5
        )
    with col2:
        show_skeleton = st.checkbox("Show Skeleton", True)
        show_angles = st.checkbox("Show Angles", True)
        show_guide = st.checkbox("Show Guide Area", True)

# 메인 컨트롤
col1, col2, col3, col4 = st.columns(4)

calibration_button = col1.button(
    "🧭 Calibration Guide",
    help="Calibrate your initial squat positions",
    use_container_width=True
)

recalibrate_button = col1.button(
    "🔄 Recalibrate Position",
    help="Recalibrate your current position",
    use_container_width=True
)

camera_check_button = col2.button(
    "📹 Camera Check",
    help="Check if your landmarks are visible to the camera",
    use_container_width=True
)

start_workout_button = col3.button(
    "🚀 Start Workout",
    help="Start real-time squat analysis",
    use_container_width=True
)

pause_button = col4.button(
    "⏸️ Pause",
    help="Pause analysis",
    use_container_width=True
)

reset_button = col4.button(
    "🔄 Reset",
    help="Reset all metrics",
    use_container_width=True
)

# 버튼 동작
analyzer = SquatAnalysisEngine()  # SquatAnalysisEngine 인스턴스 생성

if calibration_button:
    analyzer.calibration_mode = True
    analyzer.calibration_count = 0
    st.info("Calibration mode activated. Please perform 5 squats to set the baseline.")

if recalibrate_button:
    analyzer.calibration_mode = True
    analyzer.calibration_count = 0
    st.info("Recalibration mode activated. Please perform 5 squats to set the baseline.")

if camera_check_button:
    st.info("Checking camera... Please ensure your full body is visible to the camera.")
    # 카메라를 통해 랜드마크를 시각적으로 확인할 수 있는 코드를 추가할 수 있습니다.

if start_workout_button:
    st.session_state.session_active = True
    st.info("Starting workout. Good luck!")
    # 실시간 스쿼트 분석을 실행하는 코드를 추가

if pause_button:
    st.session_state.session_active = False
    st.info("Workout paused.")

if reset_button:
    analyzer.reset()
    st.session_state.session_active = False
    st.info("Workout reset. All metrics have been cleared.")

# 나머지 함수 및 클래스를 그대로 유지
# (예: SquatDataManager, PerformanceVisualizer, run_squat_analysis_system 등)

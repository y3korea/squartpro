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

# 페이지 설정
st.set_page_config(
    page_title="AI SmartSquat Pro",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-metric {
        color: #ff0000;
    }
    .success-metric {
        color: #00ff00;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    div.block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'calibration_mode' not in st.session_state:
    st.session_state.calibration_mode = False
if 'calibration_count' not in st.session_state:
    st.session_state.calibration_count = 0
if 'rep_count' not in st.session_state:
    st.session_state.rep_count = 0

@dataclass
class SquatStandards:
    """스쿼트 표준 각도 및 허용 범위 정의"""
    STANDING = {
        'hip_angle': 170,    # 기립 시 엉덩이 각도
        'knee_angle': 175,   # 기립 시 무릎 각도
        'ankle_angle': 80,   # 기립 시 발목 각도
        'tolerance': 5       # 허용 오차
    }
    
    PARALLEL = {
        'hip_angle': 95,     # 스쿼트 시 엉덩이 각도
        'knee_angle': 90,    # 스쿼트 시 무릎 각도
        'ankle_angle': 70,   # 스쿼트 시 발목 각도
        'tolerance': 10      # 허용 오차
    }

class SquatAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.standards = SquatStandards()
        self.current_phase = "STANDING"
        self.csv_folder = "squat_data"
        if not os.path.exists(self.csv_folder):
            os.makedirs(self.csv_folder)
        self.csv_file = os.path.join(self.csv_folder, "landmarks.csv")
        self._initialize_csv()

    def _initialize_csv(self):
        """CSV 파일 초기화"""
        with open(self.csv_file, mode='w') as file:
            file.write("timestamp,landmark_index,x,y,z\n")

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """프레임 분석 및 피드백 생성"""
        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        feedback = {
            'is_correct': False,
            'phase': self.current_phase,
            'angles': {},
            'feedback': '',
            'score': 0,
            'landmarks': results.pose_landmarks.landmark if results.pose_landmarks else None
        }

        if results.pose_landmarks:
            # 랜드마크 그리기
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # 각도 계산
            angles = self._calculate_angles(results.pose_landmarks.landmark)
            feedback['angles'] = angles
            
            # 자세 분석
            feedback.update(self._analyze_squat_phase(angles))
            
            # 랜드마크 좌표 저장
            self._save_coordinates_to_csv(results.pose_landmarks.landmark)
            
        return image, feedback
    
    def _calculate_angles(self, landmarks) -> Dict[str, float]:
        """관절 각도 계산"""
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

        # 주요 각도 계산
        hip_angle = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        )
        
        knee_angle = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        
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
    
    def _analyze_squat_phase(self, angles: Dict[str, float]) -> Dict:
        """스쿼트 자세 분석"""
        result = {
            'is_correct': False,
            'feedback': '',
            'score': 0
        }

        # 자세 판단
        if angles['hip'] > 160:  # 서있는 자세
            self.current_phase = "STANDING"
            result['feedback'] = "준비 자세입니다. 스쿼트를 시작하세요."
        elif angles['hip'] < 100:  # 스쿼트 자세
            self.current_phase = "SQUAT"
            
            # 각도 편차 계산
            hip_dev = abs(angles['hip'] - self.standards.PARALLEL['hip_angle'])
            knee_dev = abs(angles['knee'] - self.standards.PARALLEL['knee_angle'])
            ankle_dev = abs(angles['ankle'] - self.standards.PARALLEL['ankle_angle'])
            
            # 점수 계산 (100점 만점)
            score = max(0, 100 - (hip_dev + knee_dev + ankle_dev))
            result['score'] = score
            
            # 피드백 생성
            if score >= 90:
                result['is_correct'] = True
                result['feedback'] = "완벽한 자세입니다! 유지하세요."
            elif score >= 70:
                result['feedback'] = "좋습니다! 깊이와 자세를 조금 더 신경써주세요."
            else:
                if hip_dev > self.standards.PARALLEL['tolerance']:
                    result['feedback'] += "엉덩이 각도를 조정하세요. "
                if knee_dev > self.standards.PARALLEL['tolerance']:
                    result['feedback'] += "무릎 위치를 확인하세요. "
                if ankle_dev > self.standards.PARALLEL['tolerance']:
                    result['feedback'] += "발목 가동성을 개선하세요. "
        
        return result

    def _save_coordinates_to_csv(self, landmarks):
        """랜드마크 좌표 CSV로 저장"""
        with open(self.csv_file, mode='a') as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for idx, landmark in enumerate(landmarks):
                file.write(f"{timestamp},{idx},{landmark.x},{landmark.y},{landmark.z}\n")


def main():
    st.title("AI SmartSquat Pro")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("설정")
        target_reps = st.number_input("목표 반복 횟수", 1, 20, 5)
        detection_confidence = st.slider("인식 감도", 0.0, 1.0, 0.5)
        show_skeleton = st.checkbox("스켈레톤 표시", True)
        show_angles = st.checkbox("각도 표시", True)

    # 메인 컨트롤
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎯 보정 가이드"):
            st.session_state.calibration_mode = True
            st.session_state.calibration_count = 0
            st.info("보정 모드가 활성화되었습니다. 기준 설정을 위해 5회 스쿼트를 수행하세요.")
    
    with col2:
        if st.button("📷 카메라 체크"):
            st.info("카메라를 확인합니다. 전신이 잘 보이도록 위치를 조정하세요.")
    
    with col3:
        if st.button("▶️ 운동 시작"):
            if st.session_state.calibration_count >= 5 or not st.session_state.calibration_mode:
                st.session_state.session_active = True
                st.success("운동을 시작합니다. 화이팅!")
            else:
                st.warning("먼저 보정을 완료해주세요.")
    
    with col4:
        if st.button("⏸️ 일시정지"):
            st.session_state.session_active = False
            st.info("운동이 일시정지되었습니다.")
        
        if st.button("🔄 리셋"):
            st.session_state.session_active = False
            st.session_state.calibration_mode = False
            st.session_state.calibration_count = 0
            st.session_state.rep_count = 0
            st.info("모든 데이터가 초기화되었습니다.")

    # 비디오 캡처
    cap = cv2.VideoCapture(0)
    analyzer = SquatAnalyzer()
    
    # 메인 화면
    stframe = st.empty()
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    while st.session_state.session_active:
        ret, frame = cap.read()
        if not ret:
            st.error("카메라를 찾을 수 없습니다.")
            break
            
        # 프레임 분석
        image, feedback = analyzer.analyze_frame(frame)
        
        # 메트릭 표시
        with metrics_col1:
            st.metric("반복 횟수", st.session_state.rep_count)
        with metrics_col2:
            st.metric("현재 점수", f"{feedback['score']:.1f}")
        with metrics_col3:
            st.metric("자세", feedback['phase'])
        
        # 피드백 표시
        if feedback['feedback']:
            st.info(feedback['feedback'])
        
        # 이미지 표시
        stframe.image(image, channels="BGR", use_column_width=True)
        
        if st.session_state.calibration_mode and feedback['is_correct']:
            st.session_state.calibration_count += 1
            if st.session_state.calibration_count >= 5:
                st.session_state.calibration_mode = False
                st.success("보정이 완료되었습니다!")
        
        # 반복 횟수 카운팅
        if feedback['is_correct'] and feedback['phase'] == "SQUAT":
            if not hasattr(st.session_state, 'last_rep_time') or \
               (datetime.now() - st.session_state.last_rep_time).total_seconds() > 1.5:
                st.session_state.rep_count += 1
                st.session_state.last_rep_time = datetime.now()

    cap.release()

if __name__ == "__main__":
    main()
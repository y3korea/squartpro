import os
import shutil
from pathlib import Path

def create_directory_structure():
    """프로젝트 디렉토리 구조 생성"""
    base_path = Path("/Users/y3korea/Documents/pose_sample_test3")
    
    # 기본 디렉토리 구조
    directories = [
        ".streamlit",
        "pages",
        "src",
        "static",
        "tests",
    ]
    
    # 디렉토리 생성
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")

def create_streamlit_config():
    """Streamlit 설정 파일 생성"""
    config_content = """[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
enableCORS = false
enableXsrfProtection = true

[browser]
serverAddress = "localhost"
serverPort = 8501"""

    config_path = Path("/Users/y3korea/Documents/pose_sample_test3/.streamlit/config.toml")
    config_path.write_text(config_content)
    print(f"Created Streamlit config: {config_path}")

def create_requirements():
    """requirements.txt 파일 생성"""
    requirements = """mediapipe==0.10.13
numpy==1.26.0
opencv-contrib-python==4.10.0.84
opencv-python-headless==4.8.0.76
Pillow==9.5.0
pyttsx3==2.98
streamlit==1.25.0
python-dotenv>=0.19.0"""

    req_path = Path("/Users/y3korea/Documents/pose_sample_test3/requirements.txt")
    req_path.write_text(requirements)
    print(f"Created requirements.txt: {req_path}")

def create_main_app():
    """메인 앱 파일 생성"""
    app_content = '''import streamlit as st
from src.pose_detector import PoseDetector
from src.squat_analysis import SquatAnalysisEngine
from src.voice_feedback import VoiceFeedback

# 페이지 설정
st.set_page_config(
    page_title="AI SmartSquat Pro",
    page_icon="🏋️‍♂️",
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
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("AI SmartSquat Pro 🏋️‍♂️")
    
    # 세션 상태 초기화
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SquatAnalysisEngine()
    
    # 사이드바 설정
    with st.sidebar:
        st.header("Settings ⚙️")
        show_skeleton = st.checkbox("Show Skeleton 🦴", True)
        show_guide = st.checkbox("Show Guide Area 🎯", True)
        show_angles = st.checkbox("Show Joint Angles 📐", True)

    # 메인 컨트롤
    cols = st.columns(4)
    
    if cols[0].button("🎯 Start Calibration", use_container_width=True):
        st.session_state.analyzer = SquatAnalysisEngine()
        st.info("Please perform 5 squats to set your baseline.")
    
    if cols[1].button("📹 Camera Check", use_container_width=True):
        st.session_state.camera_check = True
    
    if cols[2].button("🚀 Start Workout", use_container_width=True):
        st.session_state.workout_active = True
    
    if cols[3].button("⏹️ Stop", use_container_width=True):
        if hasattr(st.session_state.analyzer, 'cleanup'):
            st.session_state.analyzer.cleanup()
        st.session_state.workout_active = False

if __name__ == "__main__":
    main()
'''
    
    app_path = Path("/Users/y3korea/Documents/pose_sample_test3/app.py")
    app_path.write_text(app_content)
    print(f"Created main app: {app_path}")

def create_source_files():
    """소스 코드 파일들 생성"""
    src_files = {
        "pose_detector.py": """import cv2
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
""",

        "squat_analysis.py": """import numpy as np
from dataclasses import dataclass

@dataclass
class SquatStandards:
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

class SquatAnalysisEngine:
    def __init__(self):
        self.standards = SquatStandards()
        self.rep_count = 0
        self.current_phase = "STANDING"
    
    def analyze_frame(self, landmarks):
        # 구현 예정
        pass
""",

        "voice_feedback.py": """import pyttsx3
import threading

class VoiceFeedback:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Failed to initialize voice engine: {str(e)}")
            self.engine = None
            
        self.speaking_thread = None
        self.last_feedback = ""
    
    def speak(self, text: str):
        if not self.engine or text == self.last_feedback:
            return
            
        self.last_feedback = text
        
        if self.speaking_thread and self.speaking_thread.is_alive():
            return
            
        self.speaking_thread = threading.Thread(
            target=lambda: self.engine.say(text) or self.engine.runAndWait()
        )
        self.speaking_thread.start()
"""
    }

    for filename, content in src_files.items():
        file_path = Path("/Users/y3korea/Documents/pose_sample_test3/src") / filename
        file_path.write_text(content)
        print(f"Created source file: {file_path}")

def create_pages():
    """페이지 파일들 생성"""
    pages = {
        "01_workout.py": """import streamlit as st
from src.pose_detector import PoseDetector

def show_workout_page():
    st.title("Workout Session 🏋️‍♂️")
    # 구현 예정
""",

        "02_history.py": """import streamlit as st
import pandas as pd

def show_history_page():
    st.title("Workout History 📊")
    # 구현 예정
""",

        "03_settings.py": """import streamlit as st

def show_settings_page():
    st.title("Settings ⚙️")
    # 구현 예정
"""
    }

    for filename, content in pages.items():
        file_path = Path("/Users/y3korea/Documents/pose_sample_test3/pages") / filename
        file_path.write_text(content)
        print(f"Created page file: {file_path}")

def main():
    """메인 실행 함수"""
    try:
        print("Starting project setup...")
        create_directory_structure()
        create_streamlit_config()
        create_requirements()
        create_main_app()
        create_source_files()
        create_pages()
        print("\nProject setup completed successfully! 🎉")
        print("\nTo run the project:")
        print("1. cd /Users/y3korea/Documents/pose_sample_test3")
        print("2. pip install -r requirements.txt")
        print("3. streamlit run app.py")
    except Exception as e:
        print(f"Error during setup: {str(e)}")

if __name__ == "__main__":
    main()
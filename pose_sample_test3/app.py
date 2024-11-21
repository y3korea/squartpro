import streamlit as st
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

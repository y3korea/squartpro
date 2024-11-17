import streamlit as st
import cv2
import numpy as np
from src.squat_analysis import EnhancedSquatAnalysisEngine

def main():
    st.set_page_config(
        page_title="AI SmartSquat Pro",
        page_icon="🏋️‍♂️",
        layout="wide"
    )
    
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

        # 설정 값을 분석 엔진에 전달
        if 'analyzer' in st.session_state:
            st.session_state.analyzer.show_skeleton = show_skeleton
            st.session_state.analyzer.show_guide = show_guide
            st.session_state.analyzer.show_angles = show_angles

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

    # 이미지 캡처 및 분석
    try:
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
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure you have allowed camera access and try again.")

if __name__ == "__main__":
    main()
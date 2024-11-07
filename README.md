# squartpro
스쿼트 코치
# AI SmartSquat Pro 🏋️‍♂️

AI SmartSquat Pro is an advanced squat analysis and feedback system built using Streamlit, Mediapipe, and OpenCV. It provides real-time squat form analysis and voice feedback to help users perform exercises with correct form, enhancing their workout experience and safety.

## Features
- Real-time squat form analysis and feedback
- Voice feedback for guiding squat posture and corrections
- Skeleton visualization and joint angle calculation
- Customizable calibration for personalized workout guidance
- Intuitive user interface with live video feed and feedback display

## How It Works
1. **Calibration**: Perform 5 squats to establish baseline angles and target zones.
2. **Workout**: Analyze squat form in real-time, receiving immediate feedback on posture, joint angles, and completion status.
3. **Voice Feedback**: Provides auditory guidance, signaling correct form or suggesting corrections as necessary.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/y3korea/squartpro.git
   cd squartpro

Requirements

	•	Python 3.7 or higher
	•	Webcam for video feed
	•	Speakers or headphones for voice feedback

Usage

	•	Start Calibration: Begin by performing 5 squats to set your baseline.
	•	Camera Check: Verify your camera feed is working.
	•	Start Workout: Begin the workout, and the application will monitor your squat form.
	•	Stop: End the workout session and release resources.

File Structure

	•	app.py: Main application code with Streamlit setup.
	•	requirements.txt: List of dependencies for the project.
	•	README.md: Project documentation.

Dependencies

	•	streamlit: For building the web application interface.
	•	mediapipe: For pose detection and skeletal tracking.
	•	opencv-python-headless: For image processing.
	•	numpy: For mathematical operations.
	•	pyttsx3: For text-to-speech feedback.
	•	dataclasses: For structured data representation (Python 3.7+).

Future Enhancements

	•	Adding rep counting accuracy improvements
	•	Incorporating additional exercise support
	•	Enhancing the feedback algorithm for detailed correction suggestions

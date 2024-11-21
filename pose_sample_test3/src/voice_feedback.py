import pyttsx3
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

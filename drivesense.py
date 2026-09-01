"""
DriveSense: AI-Driven In-Vehicle Digital Voice Assistant
=========================================================
Minimalist Architecture with Google Gemini.
"""

import os
import sys
import json
import time
import random
import speech_recognition as sr
import google.generativeai as genai
from typing import Dict, Any

# Ensure Windows terminal can print unicode emojis without crashing
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    # pyrefly: ignore [missing-import]
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

VALID_DOORS = ["LEFT", "RIGHT", "FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT", "REAR_RIGHT"]

class VehicleState:
    """Centralized vehicle state tracker."""
    def __init__(self):
        self.doors = {door: "CLOSED" for door in VALID_DOORS}
        self.ac_on = False
        self.last_temperature = 26.0
        self.last_distance = 50.0

class VehicleController:
    """Simulated hardware controller."""
    def __init__(self, state: VehicleState):
        self.state = state

    def get_temperature(self) -> float:
        base = 27.0 if not self.state.ac_on else 23.0
        self.state.last_temperature = round(base + random.uniform(-2.0, 3.0), 1)
        return self.state.last_temperature

    def get_distance(self) -> float:
        self.state.last_distance = round(random.uniform(15.0, 80.0), 1)
        return self.state.last_distance

    def set_door(self, target: str, open_it: bool) -> str:
        target = target.upper().strip()
        targets = VALID_DOORS if target == "ALL" else ([target] if target in VALID_DOORS else [])
        if not targets:
            return f"Unknown door: {target}."
        
        for d in targets:
            self.state.doors[d] = "OPEN" if open_it else "CLOSED"
            
        action = "Opening" if open_it else "Closing"
        name = "all" if target == "ALL" else target.replace("_", " ").lower()
        return f"{action} {name} doors."

    def set_ac(self, on: bool) -> str:
        self.state.ac_on = on
        action = "Turning on" if on else "Turning off"
        return f"{action} the AC."

class DriveSenseAI:
    """Core AI Layer using Google Gemini."""
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key and os.path.exists("api_key.txt"):
            with open("api_key.txt", "r") as f:
                api_key = f.read().strip()
                
        if not api_key:
            print("\n[!] WARNING: API key is not set. The AI will not function.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-3.6-flash')
        
        self.system_prompt = """
        You are DriveSense, an in-car AI assistant. 
        Identify the user's intent and output strictly JSON.
        
        Intents: OPEN_DOOR, CLOSE_DOOR, TURN_AC_ON, TURN_AC_OFF, GET_AC_STATUS, 
        GET_TEMPERATURE, GET_DISTANCE, GET_SENSOR_STATUS, GENERAL_QUERY, STOP_LISTENING, EXIT, UNSUPPORTED.
        
        Format: {"intent": "...", "target": "...", "response": "..."}
        """

    def generate_response(self, text: str) -> Dict[str, Any]:
        try:
            response = self.model.generate_content(
                f"{self.system_prompt}\nUser: '{text}'\nJSON:",
                request_options={"timeout": 15.0}
            )
            raw = response.text.strip()
            if raw.startswith("```json"): raw = raw[7:]
            if raw.startswith("```"): raw = raw[3:]
            if raw.endswith("```"): raw = raw[:-3]
                
            return json.loads(raw.strip())
        except Exception as e:
            print(f"\n[❌] AI Error: {e}")
            return {"intent": "UNSUPPORTED", "response": "I'm having trouble thinking right now."}

class CommandDispatcher:
    """Routes AI intent to hardware actions."""
    def __init__(self, hw: VehicleController, state: VehicleState):
        self.hw = hw
        self.state = state

    def dispatch(self, payload: Dict[str, Any]) -> str:
        intent = payload.get("intent")
        
        if intent == "GET_TEMPERATURE":
            return f"The cabin temperature is {self.hw.get_temperature()} degrees Celsius."
        if intent == "GET_DISTANCE":
            dist = self.hw.get_distance()
            return f"Warning: Obstacle {dist} cm ahead!" if dist < 30 else f"Nearest object is {dist} cm ahead."
        if intent == "GET_SENSOR_STATUS":
            return "An obstacle is detected." if self.hw.get_distance() < 30 else "Path is clear."
        if intent == "OPEN_DOOR":
            return self.hw.set_door(payload.get("target", "LEFT"), open_it=True)
        if intent == "CLOSE_DOOR":
            return self.hw.set_door(payload.get("target", "ALL"), open_it=False)
        if intent == "TURN_AC_ON":
            return self.hw.set_ac(True)
        if intent == "TURN_AC_OFF":
            return self.hw.set_ac(False)
        if intent == "GET_AC_STATUS":
            return "The AC is on." if self.state.ac_on else "The AC is off."
            
        return payload.get("response", "Done.")

class DriveSenseVoiceAssistant:
    """Main continuous voice loop."""
    def __init__(self):
        self.state = VehicleState()
        self.hw = VehicleController(self.state)
        self.ai = DriveSenseAI()
        self.dispatcher = CommandDispatcher(self.hw, self.state)
        self.is_active = False
        self.is_speaking = False

    def speak(self, text: str):
        self.is_speaking = True
        print(f"\n[🔊 DriveSense]: \"{text}\"")
        if HAS_TTS:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 170)
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
        time.sleep(0.2)
        self.is_speaking = False

    def run(self):
        print("=" * 50 + "\n   🚗 DRIVESENSE (LITE) \n" + "=" * 50)
        self.speak("DriveSense is online. Calibrating microphone...")
        
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True

        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=2.0)
                self.speak("Ready. Say DriveSense to begin.")

                while True:
                    if self.is_speaking:
                        time.sleep(0.1)
                        continue

                    mode = "🎤 ACTIVE" if self.is_active else "🔒 IDLE"
                    print(f"\r  [{mode}] Listening...               ", end="", flush=True)

                    try:
                        audio = recognizer.listen(source, timeout=1.0, phrase_time_limit=10.0)
                        print("\r  [⚡] Processing...                 ", end="", flush=True)
                        text = recognizer.recognize_google(audio).lower().strip()
                        
                        if not text: continue
                        
                        if not self.is_active:
                            if "drivesense" in text.replace(" ", ""):
                                self.is_active = True
                                print("\n✅ [WAKE WORD DETECTED]")
                                self.speak("Yes?")
                            continue

                        print(f"\n\n⚡ [COMMAND RECEIVED]: \"{text}\"")
                        payload = self.ai.generate_response(text)
                        print("--- [AI JSON Output] ---")
                        print(json.dumps(payload, indent=2))
                        
                        intent = payload.get("intent", "UNSUPPORTED")
                        if intent == "STOP_LISTENING":
                            self.speak(payload.get("response", "Going to sleep."))
                            self.is_active = False
                        elif intent == "EXIT":
                            self.speak(payload.get("response", "Goodbye."))
                            break
                        else:
                            self.speak(self.dispatcher.dispatch(payload))
                            
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        print(f"\n[❌] API Error: {e}")
                        time.sleep(2)
                        
        except KeyboardInterrupt:
            self.speak("Shutting down.")

if __name__ == "__main__":
    DriveSenseVoiceAssistant().run()

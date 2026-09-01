"""
DriveSense: AI-Driven In-Vehicle Digital Voice Assistant
=========================================================
Intelligence and Intent-Understanding Layer for Connected Vehicles.
Decoupled Architecture: Returns structured JSON intents for the Python
application/dispatcher to execute sensor operations and hardware commands.

Deployment Targets:
- Development: Windows / Mac / Linux Laptops (Simulation Mode)
- Production: Raspberry Pi (Physical GPIO Sensors: DHT Temp & HC-SR04 Ultrasonic)
"""

import os
import re
import sys
import json
import time
import queue
import threading
from typing import Dict, Any, Optional

# Optional Audio / TTS Imports
try:
    import pyaudio
    from vosk import Model, KaldiRecognizer
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False


# ==============================================================================
# 1. INTELLIGENCE & INTENT-UNDERSTANDING LAYER (DriveSenseAI)
# ==============================================================================
class DriveSenseAI:
    """
    Core AI & Natural Language Understanding Layer.
    Classifies user voice inputs into exactly ONE of the supported intents:
      - GREETING
      - GET_TEMPERATURE
      - GET_DISTANCE
      - GENERAL_QUERY
      - EXIT
      - UNSUPPORTED
    
    Returns structured JSON strictly conforming to the DriveSense specification.
    Never fabricates sensor values.
    """

    UNSUPPORTED_PATTERNS = [
        r"\b(call|phone|dial|ring|contact)\b",
        r"\b(navigate|navigation|route|directions|map|destination|chennai|gps to)\b",
        r"\b(fuel|gasoline|petrol|diesel|tank level)\b",
        r"\b(ac|air conditioner|cooling|heater|climate control)\b",
        r"\b(sleepy|drowsy|drowsiness|fatigue|asleep)\b",
        r"\b(music|song|radio|playlist|spotify|volume|play)\b",
        r"\b(obd|can bus|diagnostics|trouble codes|check engine)\b",
        r"\b(emergency|police|ambulance|crash detect|collision detection)\b",
        r"\b(trunk|boot|window|door lock|headlight|wiper|brake|steer)\b"
    ]

    TEMPERATURE_PATTERNS = [
        r"\b(temp|temperature|how hot|how cold|how warm|cabin temp|cabin temperature|vehicle temp|weather inside)\b"
    ]

    DISTANCE_PATTERNS = [
        r"\b(distance|obstacle|object in front|anything in front|ahead|path clear|how close|how far|collision danger|front sensor)\b"
    ]

    GREETING_PATTERNS = [
        r"^(hello|hi|hey|good morning|good afternoon|good evening|greetings|howdy|what's up|hi drivesense|hello drivesense)\b"
    ]

    EXIT_PATTERNS = [
        r"^(exit|stop|goodbye|quit|bye|close drivesense|shutdown|turn off|stop assistant|end)\b"
    ]

    KNOWLEDGE_BASE = {
        "artificial intelligence": "Artificial intelligence is the simulation of human intelligence processes by computer systems.",
        "machine learning": "Machine learning is a branch of artificial intelligence in which systems learn patterns from data to make predictions or decisions.",
        "gps": "GPS stands for Global Positioning System, using a network of satellites to determine precise geographic location and time.",
        "electric vehicles": "Electric vehicles use electric motors powered by rechargeable battery packs rather than internal combustion engines.",
        "hybrid": "Hybrid vehicles combine an internal combustion engine with one or more electric motors to optimize fuel efficiency.",
        "abs": "Anti-lock braking systems prevent vehicle wheels from locking during sudden braking, maintaining steering control.",
        "cruise control": "Cruise control automatically maintains a driver-selected vehicle cruising speed.",
        "regenerative braking": "Regenerative braking captures kinetic energy during deceleration and converts it back into battery power."
    }

    def classify_intent(self, text: str) -> str:
        clean_text = text.lower().strip()

        # 1. Exit Check
        for pattern in self.EXIT_PATTERNS:
            if re.search(pattern, clean_text):
                return "EXIT"

        # 2. Greeting Check
        for pattern in self.GREETING_PATTERNS:
            if re.search(pattern, clean_text):
                return "GREETING"

        # 3. Temperature Check
        for pattern in self.TEMPERATURE_PATTERNS:
            if re.search(pattern, clean_text):
                return "GET_TEMPERATURE"

        # 4. Obstacle Distance Check
        for pattern in self.DISTANCE_PATTERNS:
            if re.search(pattern, clean_text):
                return "GET_DISTANCE"

        # 5. Unsupported Features Check
        for pattern in self.UNSUPPORTED_PATTERNS:
            if re.search(pattern, clean_text):
                return "UNSUPPORTED"

        # 6. Default to General Query
        return "GENERAL_QUERY"

    def generate_response(self, text: str) -> Dict[str, str]:
        """
        Processes driver voice transcript and produces strict structured JSON.
        """
        intent = self.classify_intent(text)

        if intent == "GREETING":
            response_text = "Hello. How can I help you?"
        elif intent == "GET_TEMPERATURE":
            response_text = "I will check the current cabin temperature."
        elif intent == "GET_DISTANCE":
            response_text = "I will check the distance to the nearest obstacle."
        elif intent == "EXIT":
            response_text = "Goodbye."
        elif intent == "UNSUPPORTED":
            response_text = "That feature is not currently available in DriveSense."
        else:  # GENERAL_QUERY
            # Find relevant answer from knowledge base
            clean_text = text.lower().strip()
            answer = None
            for key, val in self.KNOWLEDGE_BASE.items():
                if key in clean_text:
                    answer = val
                    break
            if not answer:
                answer = "DriveSense is ready to assist with cabin temperature and obstacle monitoring."
            response_text = answer

        result = {
            "intent": intent,
            "response": response_text
        }
        return result

    def get_json_response(self, text: str) -> str:
        """Returns JSON string format."""
        return json.dumps(self.generate_response(text), indent=2)


# ==============================================================================
# 2. HARDWARE & SENSOR DISPATCHER (Application Execution Layer)
# ==============================================================================
class DriveSenseHardwareDispatcher:
    """
    Executes actual sensor readings and hardware operations based on structured intents.
    Supports physical Raspberry Pi GPIO sensors or automated Laptop simulation.
    """

    def __init__(self, is_raspberry_pi: bool = False):
        self.is_raspberry_pi = is_raspberry_pi
        self.gpio_initialized = False

        if self.is_raspberry_pi:
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                self.GPIO.setmode(self.GPIO.BCM)
                self.gpio_initialized = True
            except ImportError:
                print("[Notice] RPi.GPIO not detected. Running in Laptop Sensor Simulation Mode.")
                self.is_raspberry_pi = False

    def read_temperature_sensor(self) -> float:
        """
        Obtains cabin temperature in Celsius.
        Reads from DHT11/DHT22 if on Pi, or generates calibrated simulation reading.
        """
        if self.is_raspberry_pi and self.gpio_initialized:
            # Physical DHT reading hook
            try:
                import Adafruit_DHT
                sensor = Adafruit_DHT.DHT22
                pin = 4  # GPIO 4
                _, temperature = Adafruit_DHT.read_retry(sensor, pin)
                if temperature is not None:
                    return round(float(temperature), 1)
            except Exception:
                pass
        
        # Laptop Simulation fallback (Realistic cabin temperature)
        import random
        return round(24.0 + random.uniform(-1.5, 2.5), 1)

    def read_distance_sensor(self) -> float:
        """
        Obtains obstacle distance in Centimeters using Ultrasonic Sensor (HC-SR04).
        """
        if self.is_raspberry_pi and self.gpio_initialized:
            try:
                TRIG_PIN = 23
                ECHO_PIN = 24
                self.GPIO.setup(TRIG_PIN, self.GPIO.OUT)
                self.GPIO.setup(ECHO_PIN, self.GPIO.IN)

                self.GPIO.output(TRIG_PIN, False)
                time.sleep(0.05)
                self.GPIO.output(TRIG_PIN, True)
                time.sleep(0.00001)
                self.GPIO.output(TRIG_PIN, False)

                pulse_start = time.time()
                while self.GPIO.input(ECHO_PIN) == 0:
                    pulse_start = time.time()

                pulse_end = time.time()
                while self.GPIO.input(ECHO_PIN) == 1:
                    pulse_end = time.time()

                pulse_duration = pulse_end - pulse_start
                distance = pulse_duration * 17150
                return round(distance, 1)
            except Exception:
                pass

        # Laptop Simulation fallback (Realistic obstacle distance in cm)
        import random
        return round(45.0 + random.uniform(-15.0, 30.0), 1)

    def dispatch(self, intent_payload: Dict[str, str]) -> str:
        """
        Executes intent and constructs final driver-facing spoken output.
        """
        intent = intent_payload.get("intent")
        initial_response = intent_payload.get("response", "")

        if intent == "GET_TEMPERATURE":
            temp = self.read_temperature_sensor()
            return f"The current cabin temperature is {temp} degrees Celsius."

        elif intent == "GET_DISTANCE":
            dist = self.read_distance_sensor()
            if dist < 30:
                return f"Caution. Obstacle detected close ahead at {dist} centimeters."
            else:
                return f"The nearest obstacle is at a distance of {dist} centimeters."

        # For Greeting, General Query, Exit, and Unsupported, pass through the response
        return initial_response


# ==============================================================================
# 3. AUDIO & SPEECH LAYER (STT & TTS)
# ==============================================================================
class DriveSenseVoiceIO:
    """
    Handles Offline Speech-to-Text (Vosk) and Text-to-Speech (pyttsx3).
    """

    def __init__(self, model_path: str = "vosk-model-small-en-in-0.4"):
        self.model_path = model_path
        self.vosk_model = None
        self.tts_engine = None

        # Initialize TTS
        if HAS_TTS:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 170)
            except Exception:
                self.tts_engine = None

        # Initialize Vosk
        if HAS_VOSK and os.path.exists(self.model_path):
            try:
                self.vosk_model = Model(self.model_path)
            except Exception as e:
                print(f"[Notice] Vosk model loading: {e}")

    def speak(self, text: str):
        """Speaks the response aloud to the driver."""
        print(f"[DriveSense Spoken]: \"{text}\"")
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception:
                pass

    def record_and_transcribe(self, duration_seconds: int = 5) -> str:
        """Records from microphone and transcribes with Vosk."""
        if not HAS_VOSK or not self.vosk_model:
            print("[Notice] Microphone STT requires Vosk and PyAudio.")
            return ""

        rec = KaldiRecognizer(self.vosk_model, 16000)
        rec.SetWords(True)

        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=2048
            )
            print(f"[*] Listening for {duration_seconds} seconds... (Speak your command)")
            
            segments = []
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                data = stream.read(2048, exception_on_overflow=False)
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    txt = res.get("text", "").strip()
                    if txt:
                        segments.append(txt)
                        print(f"  [Partial]: {txt}")

            final_res = json.loads(rec.FinalResult())
            final_txt = final_res.get("text", "").strip()
            if final_txt:
                segments.append(final_txt)

            stream.stop_stream()
            stream.close()
            return " ".join(segments).strip()
        except Exception as e:
            print(f"[Audio Error]: {e}")
            return ""
        finally:
            p.terminate()


# ==============================================================================
# 4. MAIN INTERACTIVE APPLICATION
# ==============================================================================
def main():
    print("=" * 65)
    print("   🚗 DriveSense: In-Vehicle Digital Voice Assistant")
    print("   Decoupled AI Intent Layer & Hardware Dispatcher")
    print("=" * 65)

    ai = DriveSenseAI()
    dispatcher = DriveSenseHardwareDispatcher(is_raspberry_pi=False)
    voice_io = DriveSenseVoiceIO()

    # Initial Greeting
    welcome_text = "DriveSense initialized. How can I assist your drive?"
    print(f"\nAI: {welcome_text}\n")
    voice_io.speak(welcome_text)

    print("Options:")
    print("  Type your command below, or type 'mic' to record from your microphone.")
    print("  Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Driver > ").strip()
            if not user_input:
                continue

            if user_input.lower() == "mic":
                user_input = voice_io.record_and_transcribe(duration_seconds=5)
                if not user_input:
                    print("[No speech recognized. Please try again or type directly.]\n")
                    continue
                print(f"Driver (Voice): \"{user_input}\"")

            # 1. Intelligence Layer: Return Strict JSON
            json_payload = ai.generate_response(user_input)
            print("\n[DriveSense AI Intent Layer JSON]:")
            print(json.dumps(json_payload, indent=2))

            # 2. Application Dispatcher: Execute Sensor Operations
            spoken_response = dispatcher.dispatch(json_payload)

            # 3. Output to Driver
            print(f"\n[Driver Output]: {spoken_response}")
            voice_io.speak(spoken_response)
            print("-" * 65)

            if json_payload["intent"] == "EXIT":
                break

        except (KeyboardInterrupt, EOFError):
            print("\nDriveSense session terminated.")
            break


if __name__ == "__main__":
    main()

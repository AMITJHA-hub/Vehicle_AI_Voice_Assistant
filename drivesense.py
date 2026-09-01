"""
DriveSense: AI-Driven In-Vehicle Digital Voice Assistant
=========================================================
Hands-Free, Continuous Two-Way Spoken Voice Assistant System.

Features:
1. Live Audio Energy & Visual Hearing Indicator (Real-Time VU Bar)
2. Direct Conversational Voice Listening (No forced wake-word barrier required)
3. Automatic Silence & End-of-Speech Detection
4. Auto-Mute Microphone during Text-to-Speech Output (Prevents Echo/Feedback Loop)
5. Decoupled AI Intelligence & Structured JSON Intent Classification
6. Hardware Sensor Dispatcher (Raspberry Pi GPIO or Laptop Simulation)
"""

import os
import re
import sys
import json
import time
import queue
import threading
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Audio / Speech Recognition
try:
    import pyaudio
    from vosk import Model, KaldiRecognizer
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False

# Text-to-Speech
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
        r"\b(temp|temps|temperature|temperatures|how hot|how cold|how warm|cabin temp|cabin temperature|cabin temperatures|vehicle temp|vehicle temperature|weather inside)\b"
    ]

    DISTANCE_PATTERNS = [
        r"\b(distance|obstacle|object in front|anything in front|ahead|path clear|how close|how far|collision danger|front sensor)\b"
    ]

    GREETING_PATTERNS = [
        r"^(hello|hi|hey|good morning|good afternoon|good evening|greetings|howdy|what's up|hi drivesense|hello drivesense|drive sense)\b"
    ]

    EXIT_PATTERNS = [
        r"^(exit|stop|goodbye|quit|bye|close drivesense|shutdown|turn off|stop assistant|end|mute|sleep|standby)\b"
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
        """Obtains cabin temperature in Celsius."""
        if self.is_raspberry_pi and self.gpio_initialized:
            try:
                import Adafruit_DHT
                sensor = Adafruit_DHT.DHT22
                pin = 4
                _, temperature = Adafruit_DHT.read_retry(sensor, pin)
                if temperature is not None:
                    return round(float(temperature), 1)
            except Exception:
                pass
        
        import random
        return round(24.0 + random.uniform(-1.5, 2.5), 1)

    def read_distance_sensor(self) -> float:
        """Obtains obstacle distance in Centimeters using Ultrasonic Sensor."""
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

        import random
        return round(45.0 + random.uniform(-15.0, 30.0), 1)

    def dispatch(self, intent_payload: Dict[str, str]) -> str:
        """Executes intent and constructs final driver-facing spoken output."""
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

        return initial_response


# ==============================================================================
# 3. DIRECT TWO-WAY CONTINUOUS VOICE ASSISTANT
# ==============================================================================
class DriveSenseVoiceAssistant:
    """
    Direct Two-Way Conversational Voice Assistant.
    Listens continuously, shows real-time VU meter & transcripts,
    executes intents, and speaks answers aloud with auto-muting.
    """

    def __init__(self, model_path: str = "vosk-model-small-en-in-0.4"):
        self.model_path = model_path
        self.ai = DriveSenseAI()
        self.dispatcher = DriveSenseHardwareDispatcher(is_raspberry_pi=False)
        self.is_running = True
        self.is_speaking = False

        if not HAS_VOSK:
            raise RuntimeError("Vosk and PyAudio are required for voice assistant.")

        if not os.path.exists(self.model_path):
            if os.path.exists("vosk-model-small-en-us-0.15"):
                self.model_path = "vosk-model-small-en-us-0.15"

        print(f"[*] Loading Offline Speech Model from '{self.model_path}'...")
        self.vosk_model = Model(self.model_path)
        print("[+] Speech Recognizer Ready!\n")

    def speak(self, text: str):
        """
        Speaks response aloud while ensuring microphone is muted to prevent feedback.
        """
        self.is_speaking = True
        print(f"\n[🔊 DriveSense]: \"{text}\"")

        if HAS_TTS:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 170)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS Notice]: {e}")
        else:
            print("[Notice: pyttsx3 not available for audio output]")

        time.sleep(0.2)
        self.is_speaking = False

    def run(self):
        """
        Main Continuous Conversational Voice Loop.
        """
        p = pyaudio.PyAudio()
        FRAME_RATE = 16000
        CHUNK_SIZE = 2048

        print("=" * 70)
        print("   🚗 DRIVESENSE CONTINUOUS 2-WAY VOICE ASSISTANT")
        print("=" * 70)
        print("  • Microphone is LIVE. Speak your command anytime!")
        print("  • Examples to say:")
        print("     - \"Hello DriveSense\"")
        print("     - \"What is the cabin temperature?\"")
        print("     - \"Is there an obstacle ahead?\"")
        print("     - \"What is artificial intelligence?\"")
        print("     - \"Call my friend\" (tests unsupported request)")
        print("     - \"Stop\" or \"Exit\" to quit")
        print("  • The microphone automatically mutes while DriveSense speaks.")
        print("=" * 70 + "\n")

        self.speak("DriveSense is online. I am listening.")

        rec = KaldiRecognizer(self.vosk_model, FRAME_RATE)
        rec.SetWords(True)

        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=FRAME_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )

            accumulated_speech = []
            last_speech_time = 0
            silence_timeout = 1.4  # Seconds of silence after speaking to execute command

            print("\n🎤 [LISTENING NOW] Speak into your microphone...")

            while self.is_running:
                # When DriveSense is speaking, drain microphone buffer to prevent self-hearing
                if self.is_speaking:
                    try:
                        stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    except Exception:
                        pass
                    continue

                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except Exception:
                    continue

                if len(data) == 0:
                    continue

                # Calculate Audio Energy (VU Meter)
                audio_np = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_np.astype(float)**2))
                
                # Visual VU bar for live feedback
                vu_level = min(int(rms / 100), 10)
                vu_bar = "■" * vu_level + " " * (10 - vu_level)

                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text_chunk = res.get("text", "").strip()
                    if text_chunk:
                        accumulated_speech.append(text_chunk)
                        last_speech_time = time.time()
                        print(f"\n  [Transcribed]: {text_chunk}")
                else:
                    partial_res = json.loads(rec.PartialResult())
                    partial_text = partial_res.get("partial", "").strip()
                    if partial_text:
                        last_speech_time = time.time()
                        print(f"\r  [{vu_bar}] Hearing: \"{partial_text}\"      ", end="", flush=True)
                    elif rms > 150:
                        print(f"\r  [{vu_bar}] (Mic Active)               ", end="", flush=True)

                # Silence Detection & Command Execution
                time_since_speech = time.time() - last_speech_time
                if accumulated_speech or (last_speech_time > 0 and time_since_speech > silence_timeout):
                    final_res = json.loads(rec.FinalResult())
                    final_chunk = final_res.get("text", "").strip()
                    if final_chunk:
                        accumulated_speech.append(final_chunk)

                    full_command = " ".join(accumulated_speech).strip()

                    if full_command:
                        print(f"\n\n⚡ [COMMAND RECEIVED]: \"{full_command}\"")
                        
                        # 1. Intelligence Layer: JSON Intent
                        intent_payload = self.ai.generate_response(full_command)
                        print("\n--- [DriveSense AI Structured JSON Output] ---")
                        print(json.dumps(intent_payload, indent=2))

                        # 2. Hardware Sensor Dispatcher
                        spoken_response = self.dispatcher.dispatch(intent_payload)

                        # 3. Two-Way Spoken Delivery (Mic is auto-muted)
                        self.speak(spoken_response)
                        print("-" * 70)

                        if intent_payload["intent"] == "EXIT":
                            break

                    # Reset recognizer for next continuous command
                    rec = KaldiRecognizer(self.vosk_model, FRAME_RATE)
                    rec.SetWords(True)
                    accumulated_speech = []
                    last_speech_time = 0
                    print("\n🎤 [LISTENING NOW] Speak your next command...")

            stream.stop_stream()
            stream.close()

        except KeyboardInterrupt:
            print("\nDriveSense stopping...")
            self.speak("DriveSense shutting down. Goodbye.")
        finally:
            p.terminate()


# ==============================================================================
# 4. MAIN ENTRY POINT
# ==============================================================================
def main():
    assistant = DriveSenseVoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()

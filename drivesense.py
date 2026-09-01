"""
DriveSense: AI-Driven In-Vehicle Digital Voice Assistant
=========================================================
Realistic Automotive Voice Assistant with Hardware Abstraction Layer.

Architecture:
    USER VOICE → STT → AI Intent Classifier → Command Dispatcher → Hardware → Spoken Response

Features:
    1. 13-Intent Natural Language Understanding (Vehicle Commands)
    2. Hardware Abstraction Layer (Laptop Simulation ↔ Raspberry Pi GPIO)
    3. Centralized Vehicle State (Doors, AC, Sensors)
    4. Wake-Word Activation ("DriveSense") with IDLE/ACTIVE States
    5. Continuous Two-Way Voice with Auto-Mute During TTS
    6. Real-Time VU Meter & Silence Detection
    7. State-Aware Responses (queries reflect actual vehicle state)
"""

import os
import re
import sys
import json
import time

# Ensure Windows terminal can print unicode emojis without crashing
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import queue
import random
import threading
import speech_recognition as sr
import google.generativeai as genai
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum

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
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

# Hardware mode: "simulation" for laptop, "raspberry_pi" for real GPIO hardware
HARDWARE_MODE = "simulation"

# Raspberry Pi GPIO Pin Assignments (used only in raspberry_pi mode)
GPIO_PINS = {
    "DHT22_DATA": 4,        # Temperature sensor data pin
    "ULTRASONIC_TRIG": 23,  # HC-SR04 trigger pin
    "ULTRASONIC_ECHO": 24,  # HC-SR04 echo pin
    "IR_SENSOR": 17,        # IR obstacle sensor pin
    "AC_RELAY": 27,         # AC relay control pin
    "DOOR_SERVO": {         # Servo/motor pins per door
        "LEFT": 5,
        "RIGHT": 6,
        "FRONT_LEFT": 12,
        "FRONT_RIGHT": 13,
        "REAR_LEFT": 19,
        "REAR_RIGHT": 26,
    }
}

# Safety threshold: distance below this triggers a warning (in cm)
OBSTACLE_WARNING_THRESHOLD_CM = 30

# Valid door identifiers
VALID_DOORS = ["LEFT", "RIGHT", "FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT", "REAR_RIGHT"]


# ==============================================================================
# 2. VEHICLE STATE
# ==============================================================================
class DoorState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"

class VehicleState:
    """
    Centralized simulated vehicle state.
    Tracks doors, AC, and latest sensor readings.
    All hardware controllers read/write through this single state object.
    """

    def __init__(self):
        # Door states — all start CLOSED
        self.doors: Dict[str, DoorState] = {
            door: DoorState.CLOSED for door in VALID_DOORS
        }

        # AC state
        self.ac_on: bool = False

        # Latest sensor readings (updated by sensors on each read)
        self.last_temperature: float = 26.0
        self.last_distance: float = 50.0
        self.last_ir_obstacle: bool = False

    def get_door_summary(self) -> str:
        """Returns a human-readable summary of all door states."""
        open_doors = [d for d, s in self.doors.items() if s == DoorState.OPEN]
        if not open_doors:
            return "All doors are currently closed."
        elif len(open_doors) == len(VALID_DOORS):
            return "All doors are currently open."
        else:
            names = ", ".join(d.replace("_", " ").lower() for d in open_doors)
            return f"The following doors are open: {names}."


# ==============================================================================
# 3. HARDWARE ABSTRACTION — BASE CLASSES
# ==============================================================================

class TemperatureSensor(ABC):
    """Abstract base class for temperature sensor."""
    @abstractmethod
    def get_temperature(self) -> float:
        """Returns cabin temperature in degrees Celsius."""
        pass

class DistanceSensor(ABC):
    """Abstract base class for distance/ultrasonic sensor."""
    @abstractmethod
    def get_distance(self) -> float:
        """Returns distance to nearest obstacle in centimeters."""
        pass

class IRSensor(ABC):
    """Abstract base class for IR obstacle sensor."""
    @abstractmethod
    def is_obstacle_detected(self) -> bool:
        """Returns True if an obstacle is detected by the IR sensor."""
        pass

class DoorController(ABC):
    """Abstract base class for door control actuator."""
    @abstractmethod
    def open_door(self, target: str) -> Tuple[bool, str]:
        """Opens specified door(s). Returns (success, message)."""
        pass

    @abstractmethod
    def close_door(self, target: str) -> Tuple[bool, str]:
        """Closes specified door(s). Returns (success, message)."""
        pass

    @abstractmethod
    def get_status(self, target: str = None) -> str:
        """Returns door status string."""
        pass

class ACController(ABC):
    """Abstract base class for AC control."""
    @abstractmethod
    def turn_on(self) -> Tuple[bool, str]:
        """Turns on the AC. Returns (success, message)."""
        pass

    @abstractmethod
    def turn_off(self) -> Tuple[bool, str]:
        """Turns off the AC. Returns (success, message)."""
        pass

    @abstractmethod
    def get_status(self) -> str:
        """Returns AC status string."""
        pass


# ==============================================================================
# 4. SIMULATED HARDWARE IMPLEMENTATIONS (Laptop Development)
# ==============================================================================

class SimulatedTemperatureSensor(TemperatureSensor):
    """Simulates a DHT22 temperature sensor with realistic cabin readings."""

    def __init__(self, vehicle_state: VehicleState):
        self.state = vehicle_state

    def get_temperature(self) -> float:
        # Simulate realistic cabin temperature: 24°C to 30°C range
        # AC affects temperature slightly
        base = 27.0 if not self.state.ac_on else 23.0
        temp = round(base + random.uniform(-2.0, 3.0), 1)
        self.state.last_temperature = temp
        return temp


class SimulatedDistanceSensor(DistanceSensor):
    """Simulates an HC-SR04 ultrasonic sensor with realistic readings."""

    def __init__(self, vehicle_state: VehicleState):
        self.state = vehicle_state

    def get_distance(self) -> float:
        # Simulate distance: 10 cm to 120 cm range
        dist = round(random.uniform(15.0, 80.0), 1)
        self.state.last_distance = dist
        self.state.last_ir_obstacle = dist < OBSTACLE_WARNING_THRESHOLD_CM
        return dist


class SimulatedIRSensor(IRSensor):
    """Simulates an IR proximity/obstacle sensor."""

    def __init__(self, vehicle_state: VehicleState):
        self.state = vehicle_state

    def is_obstacle_detected(self) -> bool:
        # IR sensor correlates with distance — if distance is short, obstacle is present
        detected = self.state.last_distance < OBSTACLE_WARNING_THRESHOLD_CM
        self.state.last_ir_obstacle = detected
        return detected


class SimulatedDoorController(DoorController):
    """Simulates door actuators (servo/motor) by updating VehicleState."""

    def __init__(self, vehicle_state: VehicleState):
        self.state = vehicle_state

    def _get_targets(self, target: str) -> List[str]:
        """Resolves target string to list of door names."""
        target = target.upper().strip()
        if target == "ALL":
            return list(VALID_DOORS)
        if target in VALID_DOORS:
            return [target]
        return []

    def open_door(self, target: str) -> Tuple[bool, str]:
        doors = self._get_targets(target)
        if not doors:
            return False, f"Unknown door: {target}."

        for d in doors:
            self.state.doors[d] = DoorState.OPEN

        if target.upper() == "ALL":
            return True, "Opening all doors."
        else:
            name = target.replace("_", " ").lower()
            return True, f"Opening the {name} door."

    def close_door(self, target: str) -> Tuple[bool, str]:
        doors = self._get_targets(target)
        if not doors:
            return False, f"Unknown door: {target}."

        for d in doors:
            self.state.doors[d] = DoorState.CLOSED

        if target.upper() == "ALL":
            return True, "Closing all doors."
        else:
            name = target.replace("_", " ").lower()
            return True, f"Closing the {name} door."

    def get_status(self, target: str = None) -> str:
        if target and target.upper() != "ALL":
            doors = self._get_targets(target)
            if not doors:
                return f"Unknown door: {target}."
            d = doors[0]
            name = d.replace("_", " ").lower()
            state = self.state.doors[d].value.lower()
            return f"The {name} door is currently {state}."
        return self.state.get_door_summary()


class SimulatedACController(ACController):
    """Simulates AC relay control by updating VehicleState."""

    def __init__(self, vehicle_state: VehicleState):
        self.state = vehicle_state

    def turn_on(self) -> Tuple[bool, str]:
        self.state.ac_on = True
        return True, "Turning on the AC."

    def turn_off(self) -> Tuple[bool, str]:
        self.state.ac_on = False
        return True, "Turning off the AC."

    def get_status(self) -> str:
        status = "on" if self.state.ac_on else "off"
        return f"The AC is currently {status}."


# ==============================================================================
# 5. RASPBERRY PI HARDWARE IMPLEMENTATIONS (Future Deployment)
# ==============================================================================

class RaspberryPiTemperatureSensor(TemperatureSensor):
    """Reads real temperature from DHT22 sensor via GPIO."""

    def __init__(self, vehicle_state: VehicleState):
        self.state = vehicle_state
        self.pin = GPIO_PINS["DHT22_DATA"]

    def get_temperature(self) -> float:
        try:
            import Adafruit_DHT
            sensor = Adafruit_DHT.DHT22
            _, temperature = Adafruit_DHT.read_retry(sensor, self.pin)
            if temperature is not None:
                temp = round(float(temperature), 1)
                self.state.last_temperature = temp
                return temp
        except Exception as e:
            print(f"[Sensor Error] DHT22: {e}")
        # Fallback to last known reading
        return self.state.last_temperature


class RaspberryPiDistanceSensor(DistanceSensor):
    """Reads real distance from HC-SR04 ultrasonic sensor via GPIO."""

    def __init__(self, vehicle_state: VehicleState, gpio_module):
        self.state = vehicle_state
        self.GPIO = gpio_module
        self.trig = GPIO_PINS["ULTRASONIC_TRIG"]
        self.echo = GPIO_PINS["ULTRASONIC_ECHO"]
        self.GPIO.setup(self.trig, self.GPIO.OUT)
        self.GPIO.setup(self.echo, self.GPIO.IN)

    def get_distance(self) -> float:
        try:
            self.GPIO.output(self.trig, False)
            time.sleep(0.05)
            self.GPIO.output(self.trig, True)
            time.sleep(0.00001)
            self.GPIO.output(self.trig, False)

            timeout = time.time() + 0.5
            pulse_start = time.time()
            while self.GPIO.input(self.echo) == 0:
                pulse_start = time.time()
                if time.time() > timeout:
                    break

            pulse_end = time.time()
            while self.GPIO.input(self.echo) == 1:
                pulse_end = time.time()
                if time.time() > timeout:
                    break

            pulse_duration = pulse_end - pulse_start
            distance = round(pulse_duration * 17150, 1)
            self.state.last_distance = distance
            self.state.last_ir_obstacle = distance < OBSTACLE_WARNING_THRESHOLD_CM
            return distance
        except Exception as e:
            print(f"[Sensor Error] HC-SR04: {e}")
            return self.state.last_distance


class RaspberryPiIRSensor(IRSensor):
    """Reads real IR obstacle sensor via GPIO."""

    def __init__(self, vehicle_state: VehicleState, gpio_module):
        self.state = vehicle_state
        self.GPIO = gpio_module
        self.pin = GPIO_PINS["IR_SENSOR"]
        self.GPIO.setup(self.pin, self.GPIO.IN)

    def is_obstacle_detected(self) -> bool:
        try:
            # IR sensor: LOW when obstacle detected (active-low)
            detected = self.GPIO.input(self.pin) == 0
            self.state.last_ir_obstacle = detected
            return detected
        except Exception as e:
            print(f"[Sensor Error] IR: {e}")
            return self.state.last_ir_obstacle


class RaspberryPiDoorController(DoorController):
    """Controls door servos/motors via GPIO. Same interface as Simulated."""

    def __init__(self, vehicle_state: VehicleState, gpio_module):
        self.state = vehicle_state
        self.GPIO = gpio_module
        self.pins = GPIO_PINS["DOOR_SERVO"]
        # Setup all door servo pins as output
        for pin in self.pins.values():
            self.GPIO.setup(pin, self.GPIO.OUT)

    def _get_targets(self, target: str) -> List[str]:
        target = target.upper().strip()
        if target == "ALL":
            return list(VALID_DOORS)
        if target in VALID_DOORS:
            return [target]
        return []

    def _actuate_door(self, door: str, open_it: bool):
        """Send PWM signal to servo to open/close door."""
        pin = self.pins.get(door)
        if pin is None:
            return
        try:
            pwm = self.GPIO.PWM(pin, 50)  # 50Hz for servo
            pwm.start(0)
            duty = 7.5 if open_it else 2.5  # Open ~90° / Close ~0°
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.5)
            pwm.ChangeDutyCycle(0)
            pwm.stop()
        except Exception as e:
            print(f"[Actuator Error] Door servo {door}: {e}")

    def open_door(self, target: str) -> Tuple[bool, str]:
        doors = self._get_targets(target)
        if not doors:
            return False, f"Unknown door: {target}."
        for d in doors:
            self._actuate_door(d, open_it=True)
            self.state.doors[d] = DoorState.OPEN
        if target.upper() == "ALL":
            return True, "Opening all doors."
        name = target.replace("_", " ").lower()
        return True, f"Opening the {name} door."

    def close_door(self, target: str) -> Tuple[bool, str]:
        doors = self._get_targets(target)
        if not doors:
            return False, f"Unknown door: {target}."
        for d in doors:
            self._actuate_door(d, open_it=False)
            self.state.doors[d] = DoorState.CLOSED
        if target.upper() == "ALL":
            return True, "Closing all doors."
        name = target.replace("_", " ").lower()
        return True, f"Closing the {name} door."

    def get_status(self, target: str = None) -> str:
        if target and target.upper() != "ALL":
            doors = self._get_targets(target)
            if not doors:
                return f"Unknown door: {target}."
            d = doors[0]
            name = d.replace("_", " ").lower()
            state = self.state.doors[d].value.lower()
            return f"The {name} door is currently {state}."
        return self.state.get_door_summary()


class RaspberryPiACController(ACController):
    """Controls AC relay via GPIO."""

    def __init__(self, vehicle_state: VehicleState, gpio_module):
        self.state = vehicle_state
        self.GPIO = gpio_module
        self.pin = GPIO_PINS["AC_RELAY"]
        self.GPIO.setup(self.pin, self.GPIO.OUT)
        self.GPIO.output(self.pin, self.GPIO.LOW)

    def turn_on(self) -> Tuple[bool, str]:
        try:
            self.GPIO.output(self.pin, self.GPIO.HIGH)
            self.state.ac_on = True
            return True, "Turning on the AC."
        except Exception as e:
            return False, f"I couldn't turn on the AC. {e}"

    def turn_off(self) -> Tuple[bool, str]:
        try:
            self.GPIO.output(self.pin, self.GPIO.LOW)
            self.state.ac_on = False
            return True, "Turning off the AC."
        except Exception as e:
            return False, f"I couldn't turn off the AC. {e}"

    def get_status(self) -> str:
        status = "on" if self.state.ac_on else "off"
        return f"The AC is currently {status}."


# ==============================================================================
# 6. HARDWARE FACTORY
# ==============================================================================

def create_hardware(mode: str, vehicle_state: VehicleState) -> Dict[str, Any]:
    """
    Factory function that creates the correct hardware implementations
    based on HARDWARE_MODE. The rest of the application interacts only
    through the abstract interfaces — it never knows which mode is active.
    """
    if mode == "raspberry_pi":
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            print("[Hardware] Raspberry Pi GPIO initialized (BCM mode).")
            return {
                "temperature": RaspberryPiTemperatureSensor(vehicle_state),
                "distance": RaspberryPiDistanceSensor(vehicle_state, GPIO),
                "ir": RaspberryPiIRSensor(vehicle_state, GPIO),
                "door": RaspberryPiDoorController(vehicle_state, GPIO),
                "ac": RaspberryPiACController(vehicle_state, GPIO),
                "_gpio": GPIO,  # Keep reference for cleanup
            }
        except ImportError:
            print("[Hardware] RPi.GPIO not found. Falling back to Simulation Mode.")

    # Default: Simulation Mode
    print("[Hardware] Running in Laptop Simulation Mode.")
    return {
        "temperature": SimulatedTemperatureSensor(vehicle_state),
        "distance": SimulatedDistanceSensor(vehicle_state),
        "ir": SimulatedIRSensor(vehicle_state),
        "door": SimulatedDoorController(vehicle_state),
        "ac": SimulatedACController(vehicle_state),
    }


# ==============================================================================
# 7. DRIVESENSE AI — INTELLIGENCE & INTENT-UNDERSTANDING LAYER
# ==============================================================================

class DriveSenseAI:
    """
    Core AI & Natural Language Understanding Layer.
    Uses Google Gemini LLM to interpret user motives and natural language.
    Returns structured JSON strictly conforming to the DriveSense specification.
    """

    def __init__(self):
        # Read API key from local file so it is not pushed to GitHub
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key and os.path.exists("api_key.txt"):
            with open("api_key.txt", "r") as f:
                api_key = f.read().strip()
                
        if not api_key or api_key == "YOUR_API_KEY":
            print("\n[!] WARNING: API key is not set.")
            print("[!] The AI will not be able to process commands.")
        
        genai.configure(api_key=api_key)
        
        # Use gemini-3.6-flash as requested by the API
        self.model = genai.GenerativeModel('gemini-3.6-flash')
        
        self.system_prompt = """
        You are the AI brain of a car named DriveSense. The user will give you a command or state their motive/feeling.
        Your job is to understand their motive and output a strictly formatted JSON object.
        
        Supported Intents:
        - OPEN_DOOR (Target: ALL, FRONT_LEFT, FRONT_RIGHT, REAR_LEFT, REAR_RIGHT, LEFT, RIGHT)
        - CLOSE_DOOR (Target: ALL, FRONT_LEFT, FRONT_RIGHT, REAR_LEFT, REAR_RIGHT, LEFT, RIGHT)
        - TURN_AC_ON
        - TURN_AC_OFF
        - GET_AC_STATUS
        - GET_TEMPERATURE
        - GET_DISTANCE (Ultrasonic collision detection)
        - GET_SENSOR_STATUS (IR sensor detection)
        - GENERAL_QUERY (For answering general questions)
        - STOP_LISTENING
        - EXIT
        - UNSUPPORTED
        
        Examples of Motives:
        - "I'm freezing in here" -> {"intent": "TURN_AC_OFF", "response": "Turning off the AC to warm up the cabin."}
        - "I am so hot" -> {"intent": "TURN_AC_ON", "response": "Turning on the AC to cool you down."}
        - "Let some air in on my side" (Driver) -> {"intent": "OPEN_DOOR", "target": "FRONT_LEFT", "response": "Opening the front left door."}
        - "I feel claustrophobic" -> {"intent": "OPEN_DOOR", "target": "ALL", "response": "Opening all doors for you."}
        - "Are we going to hit something?" -> {"intent": "GET_DISTANCE", "response": "Checking obstacle distance."}
        - "How does a car engine work?" -> {"intent": "GENERAL_QUERY", "response": "An engine works by igniting fuel and air..."}
        
        You must ONLY output valid JSON. Do NOT wrap it in markdown code blocks.
        The JSON must contain "intent" and "response", and "target" if applicable.
        """

    def generate_response(self, text: str) -> Dict[str, Any]:
        """
        Processes driver voice input using Gemini and produces strict structured JSON.
        """
        try:
            prompt = f"{self.system_prompt}\n\nUser Input: '{text}'\nOutput JSON:"
            
            # Request timeout set to 15 seconds so it doesn't hang forever on bad networks
            response = self.model.generate_content(
                prompt,
                request_options={"timeout": 15.0}
            )
            
            # Clean up potential markdown formatting from Gemini
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            payload = json.loads(raw_text.strip())
            
            # Ensure required fields exist
            if "intent" not in payload:
                payload["intent"] = "UNSUPPORTED"
            if "response" not in payload:
                payload["response"] = "I processed the command but have no response."
                
            return payload
            
        except Exception as e:
            print(f"\n[❌] AI Processing Error: {e}")
            return {
                "intent": "UNSUPPORTED",
                "response": "I'm sorry, I am having trouble connecting to my AI brain right now."
            }


# ==============================================================================
# 8. COMMAND DISPATCHER
# ==============================================================================

class CommandDispatcher:
    """
    Routes structured AI intent payloads to the correct hardware controller.
    Executes the action and returns a state-aware spoken response string.
    The dispatcher is the ONLY component that touches hardware.
    """

    def __init__(self, hardware: Dict[str, Any], vehicle_state: VehicleState):
        self.hw = hardware
        self.state = vehicle_state

    def dispatch(self, payload: Dict[str, Any]) -> str:
        """
        Receives structured intent JSON from DriveSenseAI.
        Executes the hardware action and returns spoken response.
        """
        intent = payload.get("intent")

        # --- Temperature Sensor ---
        if intent == "GET_TEMPERATURE":
            temp = self.hw["temperature"].get_temperature()
            return f"The cabin temperature is {temp} degrees Celsius."

        # --- Distance Sensor (Ultrasonic) ---
        if intent == "GET_DISTANCE":
            dist = self.hw["distance"].get_distance()
            if dist < OBSTACLE_WARNING_THRESHOLD_CM:
                return f"Warning. An obstacle is very close, approximately {dist} centimeters ahead."
            return f"The nearest obstacle is approximately {dist} centimeters ahead."

        # --- IR Sensor Status ---
        if intent == "GET_SENSOR_STATUS":
            detected = self.hw["ir"].is_obstacle_detected()
            if detected:
                return "An obstacle is detected by the IR sensor."
            return "No obstacle is currently detected by the IR sensor."

        # --- Door Control ---
        if intent == "OPEN_DOOR":
            target = payload.get("target", "LEFT")
            success, message = self.hw["door"].open_door(target)
            if success:
                return message
            return f"I couldn't open the door. {message}"

        if intent == "CLOSE_DOOR":
            target = payload.get("target", "ALL")
            success, message = self.hw["door"].close_door(target)
            if success:
                return message
            return f"I couldn't close the door. {message}"

        # --- AC Control ---
        if intent == "TURN_AC_ON":
            success, message = self.hw["ac"].turn_on()
            if success:
                return message
            return f"I couldn't turn on the AC. {message}"

        if intent == "TURN_AC_OFF":
            success, message = self.hw["ac"].turn_off()
            if success:
                return message
            return f"I couldn't turn off the AC. {message}"

        if intent == "GET_AC_STATUS":
            return self.hw["ac"].get_status()

        # --- Non-hardware intents: return AI-generated response directly ---
        return payload.get("response", "")


# ==============================================================================
# 9. VOICE ASSISTANT — CONTINUOUS TWO-WAY CONVERSATION
# ==============================================================================

class AssistantState(Enum):
    IDLE = "IDLE"       # Waiting for wake word
    ACTIVE = "ACTIVE"   # Listening for commands

WAKE_WORDS = [
    "hello", "drivesense", "drive sense", "hey drive sense", "hey drivesense", 
    "okay drivesense", "okay drive sense", "dry sense", "drive sent", 
    "drive cents", "drive since", "hi drivesense", "assistant"
]


class DriveSenseVoiceAssistant:
    """
    Continuous Two-Way Conversational Voice Assistant with Wake-Word Activation.

    States:
        IDLE   — Mic is live but only listens for wake word "DriveSense"
        ACTIVE — Processes all voice commands and executes via dispatcher

    The mic auto-mutes during TTS to prevent echo/feedback.
    """

    def __init__(self, model_path: str = "vosk-model-en-us-0.22"):
        self.model_path = model_path
        self.ai = DriveSenseAI()
        self.assistant_state = AssistantState.IDLE
        self.is_running = True
        self.is_speaking = False

        # Initialize vehicle state and hardware
        self.vehicle_state = VehicleState()
        self.hardware = create_hardware(HARDWARE_MODE, self.vehicle_state)
        self.dispatcher = CommandDispatcher(self.hardware, self.vehicle_state)

        if not HAS_VOSK:
            raise RuntimeError("Vosk and PyAudio are required for voice assistant.")

        if not os.path.exists(self.model_path):
            if os.path.exists("vosk-model-small-en-in-0.4"):
                self.model_path = "vosk-model-small-en-in-0.4"

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

    def _is_wake_word(self, text: str) -> bool:
        """Check if the transcribed text contains the wake word, including fuzzy matches."""
        clean = text.lower().strip()
        
        # 1. Exact or substring match (fastest)
        for wake in WAKE_WORDS:
            if wake in clean:
                return True
                
        # 2. Fuzzy match to catch slight speech recognition errors
        import difflib
        words = clean.split()
        for i in range(len(words)):
            unigram = words[i]
            bigram = " ".join(words[i:i+2]) if i < len(words) - 1 else ""
            
            for wake in WAKE_WORDS:
                wake_len = len(wake.split())
                if wake_len == 1:
                    if difflib.SequenceMatcher(None, unigram, wake).ratio() > 0.85:
                        return True
                elif wake_len == 2 and bigram:
                    if difflib.SequenceMatcher(None, bigram, wake).ratio() > 0.85:
                        return True
        return False

    def _print_vehicle_state(self):
        """Print current simulated vehicle state for debugging."""
        doors_str = ", ".join(
            f"{d}: {s.value}" for d, s in self.vehicle_state.doors.items()
        )
        ac_str = "ON" if self.vehicle_state.ac_on else "OFF"
        print(f"  [Vehicle State] AC={ac_str} | Doors: {doors_str}")

    def run(self):
        """
        Main Continuous Conversational Voice Loop using SpeechRecognition library.
        Provides robust Ambient Noise Adjustment and precise Voice Activity Detection.
        """
        USE_OFFLINE_VOSK = False # Toggle this to True if you have no internet access on the Raspberry Pi
        
        print("=" * 70)
        print("   🚗 DRIVESENSE — IN-VEHICLE AI VOICE ASSISTANT")
        print("=" * 70)
        print(f"  • Hardware Mode: {HARDWARE_MODE.upper()}")
        print(f"  • Wake Word: \"DriveSense\"")
        print("  • Engine: " + ("VOSK OFFLINE" if USE_OFFLINE_VOSK else "GOOGLE ONLINE (Maximum Accuracy)"))
        print("  • Voice Commands:")
        print("     - \"DriveSense\" (wake up)")
        print("     - \"Open the left door\" / \"Close all doors\"")
        print("     - \"Turn on the AC\" / \"Turn off the AC\"")
        print("     - \"Is the AC on?\"")
        print("     - \"What is the cabin temperature?\"")
        print("     - \"Is there an obstacle ahead?\"")
        print("     - \"Stop listening\" (pause) / \"Exit\" (shutdown)")
        print("=" * 70 + "\n")

        self.speak("DriveSense is online. Please wait while I calibrate the microphone.")

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300 # Baseline
        recognizer.dynamic_energy_threshold = True

        try:
            with sr.Microphone() as source:
                print("\n[🛠️] Calibrating microphone for ambient room noise... (Please be quiet for 2 seconds)")
                recognizer.adjust_for_ambient_noise(source, duration=2.0)
                print(f"[🛠️] Calibration complete! Threshold set to: {int(recognizer.energy_threshold)}\n")

                self.speak("Calibration complete. Say DriveSense to begin.")
                print("🔒 [IDLE] Waiting for wake word \"DriveSense\"...")

                while self.is_running:
                    # Mute mic buffer while speaking by flushing it visually
                    if self.is_speaking:
                        time.sleep(0.1)
                        continue

                    # Visual feedback based on state
                    state_icon = "🔒" if self.assistant_state == AssistantState.IDLE else "🎤"
                    state_msg = "Waiting for wake word" if self.assistant_state == AssistantState.IDLE else "Speak your command"
                    print(f"\r  [{state_icon} {self.assistant_state.value}] {state_msg}...              ", end="", flush=True)

                    try:
                        # Listen for a phrase (blocks until silence is detected)
                        audio = recognizer.listen(source, timeout=1.0, phrase_time_limit=10.0)
                        
                        # Process Speech
                        print(f"\r  [⚡] Processing Speech...                                  ", end="", flush=True)
                        if USE_OFFLINE_VOSK:
                            # Requires vosk to be passed as JSON string
                            res_json = recognizer.recognize_vosk(audio)
                            res_dict = json.loads(res_json)
                            text = res_dict.get("text", "")
                        else:
                            text = recognizer.recognize_google(audio)

                        full_transcription = text.strip()

                        if not full_transcription:
                            continue

                        # ---- PROCESS TRANSCRIPTION ----
                        if self.assistant_state == AssistantState.IDLE:
                            print(f"\n  [Heard in IDLE]: {full_transcription}")
                            if self._is_wake_word(full_transcription):
                                self.assistant_state = AssistantState.ACTIVE
                                print("\n✅ [WAKE WORD DETECTED]")
                                self.speak("Yes?")
                            else:
                                print("\n🔒 [IDLE] Waiting for wake word \"DriveSense\"...")
                        
                        elif self.assistant_state == AssistantState.ACTIVE:
                            print(f"\n\n⚡ [COMMAND RECEIVED]: \"{full_transcription}\"")
                            
                            intent_payload = self.ai.generate_response(full_transcription)
                            print("\n--- [DriveSense AI Structured JSON Output] ---")
                            print(json.dumps(intent_payload, indent=2))
                            intent = intent_payload["intent"]

                            if intent == "STOP_LISTENING":
                                self.speak(intent_payload["response"])
                                self.assistant_state = AssistantState.IDLE
                                print("\n🔒 [IDLE] Waiting for wake word \"DriveSense\"...")
                            elif intent == "EXIT":
                                self.speak(intent_payload["response"])
                                break
                            else:
                                spoken_response = self.dispatcher.dispatch(intent_payload)
                                self.speak(spoken_response)
                                self._print_vehicle_state()
                                print("-" * 70)

                    except sr.WaitTimeoutError:
                        # Re-loop and update UI
                        continue
                    except sr.UnknownValueError:
                        # Recognizer didn't understand (background noise or mumbles)
                        continue
                    except sr.RequestError as e:
                        print(f"\n[❌] Could not request results from Google API; {e}")
                        if not USE_OFFLINE_VOSK:
                            print("[❌] Check your internet connection or switch USE_OFFLINE_VOSK to True.")
                            time.sleep(2)
                        continue
                    except Exception as e:
                        print(f"\n[❌] Audio Error: {e}")
                        continue

        except KeyboardInterrupt:
            print("\nDriveSense stopping...")
            self.speak("DriveSense shutting down. Goodbye.")
        finally:
            gpio = self.hardware.get("_gpio")
            if gpio:
                gpio.cleanup()


# ==============================================================================
# 10. MAIN ENTRY POINT
# ==============================================================================

def main():
    assistant = DriveSenseVoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()

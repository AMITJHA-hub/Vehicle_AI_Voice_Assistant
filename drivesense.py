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
    Classifies user voice inputs into one of 13 supported intents
    and extracts parameters (e.g., door target) when applicable.

    Supported Intents:
        GREETING, GET_TEMPERATURE, GET_DISTANCE, OPEN_DOOR, CLOSE_DOOR,
        TURN_AC_ON, TURN_AC_OFF, GET_AC_STATUS, GET_SENSOR_STATUS,
        GENERAL_QUERY, STOP_LISTENING, EXIT, UNSUPPORTED

    Returns structured JSON strictly conforming to the DriveSense specification.
    Never fabricates sensor values. Never directly controls hardware.
    """

    # --- Intent Pattern Definitions ---

    EXIT_PATTERNS = [
        r"^(exit|goodbye|quit|bye|shutdown|turn off assistant|end session)\b"
    ]

    STOP_LISTENING_PATTERNS = [
        r"\b(stop listening|go to sleep|pause listening|mute yourself|standby|wait for wake word)\b",
        r"^stop$",  # bare "stop" only when it's the entire command
    ]

    GREETING_PATTERNS = [
        r"^(hello|hi|hey|good morning|good afternoon|good evening|greetings|howdy|what's up|hi drivesense|hello drivesense)\b"
    ]

    TEMPERATURE_PATTERNS = [
        r"\b(temp|temps|temperature|temperatures|how hot|how cold|how warm|cabin temp|cabin temperature|cabin temperatures|vehicle temp|vehicle temperature|weather inside)\b"
    ]

    DISTANCE_PATTERNS = [
        r"\b(distance|obstacle|object in front|anything in front|ahead|path clear|how close|how far|collision danger|front sensor|in front of me)\b"
    ]

    # Door patterns: detect open/close + extract target door
    OPEN_DOOR_PATTERNS = [
        r"\b(open)\b.*\b(door|doors)\b",
        r"\b(door|doors)\b.*\b(open)\b",
        r"\b(unlock)\b.*\b(door|doors)\b",
    ]

    CLOSE_DOOR_PATTERNS = [
        r"\b(close|shut|lock)\b.*\b(door|doors)\b",
        r"\b(door|doors)\b.*\b(close|shut|lock)\b",
    ]

    AC_ON_PATTERNS = [
        r"\b(turn on|switch on|start|activate|enable)\b.*\b(ac|a\.c\.|air conditioner|air conditioning|cooling)\b",
        r"\b(ac|a\.c\.|air conditioner|air conditioning|cooling)\b.*\b(on|start|activate)\b",
    ]

    AC_OFF_PATTERNS = [
        r"\b(turn off|switch off|stop|deactivate|disable)\b.*\b(ac|a\.c\.|air conditioner|air conditioning|cooling)\b",
        r"\b(ac|a\.c\.|air conditioner|air conditioning|cooling)\b.*\b(off|stop|deactivate)\b",
    ]

    AC_STATUS_PATTERNS = [
        r"\b(is the|is my|check|status of|what.s the)\b.*\b(ac|a\.c\.|air conditioner|air conditioning|cooling)\b",
        r"\b(ac|a\.c\.|air conditioner|air conditioning)\b.*\b(status|running|working)\b",
    ]

    SENSOR_STATUS_PATTERNS = [
        r"\b(sensor|sensors|ir sensor|infrared)\b",
        r"\b(ir)\b.*\b(sensor|status|reading|check|detect)\b",
        r"\b(sensor|sensors)\b.*\b(status|reading|check|detect)\b",
        r"\bobstacle\b.*\bdetect",
    ]

    UNSUPPORTED_PATTERNS = [
        r"\b(call|phone|dial|ring|contact)\b",
        r"\b(navigate|navigation|route|directions|map|destination|gps to)\b",
        r"\b(fuel|gasoline|petrol|diesel|tank level)\b",
        r"\b(sleepy|drowsy|drowsiness|fatigue|asleep)\b",
        r"\b(music|song|radio|playlist|spotify|volume|play)\b",
        r"\b(obd|can bus|diagnostics|trouble codes|check engine)\b",
        r"\b(emergency|police|ambulance|crash detect)\b",
        r"\b(window|windows|headlight|headlights|wiper|wipers|brake|brakes|steer|steering|trunk|boot)\b",
        r"\b(engine|ignition|start the car|park|parking)\b",
    ]

    # Door target extraction patterns
    DOOR_TARGET_MAP = {
        r"\b(all)\b.*\b(door|doors)\b": "ALL",
        r"\b(door|doors)\b.*\b(all)\b": "ALL",
        r"\b(every)\b.*\b(door|doors)\b": "ALL",
        r"\b(front left|front.left|driver.s front)\b": "FRONT_LEFT",
        r"\b(front right|front.right|passenger.s front)\b": "FRONT_RIGHT",
        r"\b(rear left|rear.left|back left|back.left)\b": "REAR_LEFT",
        r"\b(rear right|rear.right|back right|back.right)\b": "REAR_RIGHT",
        r"\b(left)\b": "LEFT",
        r"\b(right)\b": "RIGHT",
    }

    KNOWLEDGE_BASE = {
        "artificial intelligence": "Artificial intelligence is the simulation of human intelligence processes by computer systems.",
        "machine learning": "Machine learning is a branch of artificial intelligence in which systems learn patterns from data to make predictions or decisions.",
        "gps": "GPS stands for Global Positioning System, using a network of satellites to determine precise geographic location and time.",
        "electric vehicles": "Electric vehicles use electric motors powered by rechargeable battery packs rather than internal combustion engines.",
        "hybrid": "Hybrid vehicles combine an internal combustion engine with one or more electric motors to optimize fuel efficiency.",
        "abs": "Anti-lock braking systems prevent vehicle wheels from locking during sudden braking, maintaining steering control.",
        "cruise control": "Cruise control automatically maintains a driver-selected vehicle cruising speed.",
        "regenerative braking": "Regenerative braking captures kinetic energy during deceleration and converts it back into battery power.",
        "autonomous driving": "Autonomous driving uses sensors, cameras and AI to navigate and control a vehicle without human input.",
        "lidar": "LiDAR uses laser pulses to measure distances and create 3D maps of the surrounding environment.",
        "embedded systems": "Embedded systems are specialized computing systems designed to perform dedicated functions within larger mechanical or electronic systems.",
        "raspberry pi": "Raspberry Pi is a low-cost single-board computer widely used for prototyping embedded and IoT applications.",
    }

    def _match_any(self, text: str, patterns: list) -> bool:
        """Returns True if text matches any pattern in the list."""
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def _extract_door_target(self, text: str) -> str:
        """Extracts the door target from natural language text."""
        # Check patterns in order — more specific first (front_left before left)
        for pattern, target in self.DOOR_TARGET_MAP.items():
            if re.search(pattern, text):
                return target
        # Default: if no specific door mentioned, assume ALL for close, LEFT for open
        return "LEFT"

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classifies text into a structured intent payload with parameters.
        Returns dict with 'intent' and optional 'target', 'query', 'sensor' fields.
        """
        clean = text.lower().strip()

        # 1. Exit (complete shutdown)
        if self._match_any(clean, self.EXIT_PATTERNS):
            return {"intent": "EXIT"}

        # 2. Greeting
        if self._match_any(clean, self.GREETING_PATTERNS):
            return {"intent": "GREETING"}

        # 3. Open Door (before general door/unsupported check)
        if self._match_any(clean, self.OPEN_DOOR_PATTERNS):
            target = self._extract_door_target(clean)
            return {"intent": "OPEN_DOOR", "target": target}

        # 4. Close Door
        if self._match_any(clean, self.CLOSE_DOOR_PATTERNS):
            target = self._extract_door_target(clean)
            return {"intent": "CLOSE_DOOR", "target": target}

        # 5. AC Status (MUST be checked BEFORE AC On/Off to avoid misclassification)
        if self._match_any(clean, self.AC_STATUS_PATTERNS):
            return {"intent": "GET_AC_STATUS"}

        # 6. AC On
        if self._match_any(clean, self.AC_ON_PATTERNS):
            return {"intent": "TURN_AC_ON"}

        # 7. AC Off
        if self._match_any(clean, self.AC_OFF_PATTERNS):
            return {"intent": "TURN_AC_OFF"}

        # 8. Temperature
        if self._match_any(clean, self.TEMPERATURE_PATTERNS):
            return {"intent": "GET_TEMPERATURE"}

        # 9. IR Sensor Status (check before distance to catch sensor-specific queries)
        if self._match_any(clean, self.SENSOR_STATUS_PATTERNS):
            return {"intent": "GET_SENSOR_STATUS", "sensor": "IR"}

        # 10. Distance / Obstacle (ultrasonic)
        if self._match_any(clean, self.DISTANCE_PATTERNS):
            return {"intent": "GET_DISTANCE"}

        # 11. Unsupported Features
        if self._match_any(clean, self.UNSUPPORTED_PATTERNS):
            return {"intent": "UNSUPPORTED"}

        # 12. Stop Listening (checked late so AC/door "stop" commands are caught first)
        if self._match_any(clean, self.STOP_LISTENING_PATTERNS):
            return {"intent": "STOP_LISTENING"}

        # 13. General Query (fallback)
        return {"intent": "GENERAL_QUERY", "query": text}

    def generate_response(self, text: str) -> Dict[str, Any]:
        """
        Processes driver voice input and produces strict structured JSON.
        The AI layer determines intent + parameters. It does NOT execute actions.
        """
        payload = self.classify_intent(text)
        intent = payload["intent"]

        # Attach initial response text from AI (dispatcher may override for sensor intents)
        if intent == "GREETING":
            payload["response"] = "Hello. How can I assist you?"
        elif intent == "GET_TEMPERATURE":
            payload["response"] = "Checking cabin temperature."
        elif intent == "GET_DISTANCE":
            payload["response"] = "Checking obstacle distance."
        elif intent == "OPEN_DOOR":
            target = payload.get("target", "LEFT")
            name = target.replace("_", " ").lower()
            if target == "ALL":
                payload["response"] = "Opening all doors."
            else:
                payload["response"] = f"Opening the {name} door."
        elif intent == "CLOSE_DOOR":
            target = payload.get("target", "ALL")
            name = target.replace("_", " ").lower()
            if target == "ALL":
                payload["response"] = "Closing all doors."
            else:
                payload["response"] = f"Closing the {name} door."
        elif intent == "TURN_AC_ON":
            payload["response"] = "Turning on the AC."
        elif intent == "TURN_AC_OFF":
            payload["response"] = "Turning off the AC."
        elif intent == "GET_AC_STATUS":
            payload["response"] = "Checking AC status."
        elif intent == "GET_SENSOR_STATUS":
            payload["response"] = "Checking sensor status."
        elif intent == "STOP_LISTENING":
            payload["response"] = "Okay. I'll wait for the wake word."
        elif intent == "EXIT":
            payload["response"] = "DriveSense shutting down. Goodbye."
        elif intent == "UNSUPPORTED":
            payload["response"] = "That feature is not currently available in DriveSense."
        else:  # GENERAL_QUERY
            clean_text = text.lower().strip()
            answer = None
            for key, val in self.KNOWLEDGE_BASE.items():
                if key in clean_text:
                    answer = val
                    break
            if not answer:
                answer = "I can help with cabin temperature, obstacle detection, door control, and AC control. Try asking about one of those."
            payload["response"] = answer

        return payload


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
        Main Continuous Conversational Voice Loop with VAD State Machine.
        Stages:
          1. WAITING_FOR_SOUND: Monitor RMS, skip Vosk processing (saves CPU/memory).
          2. LISTENING: Feed audio to Vosk until silence timeout is reached.
          3. PROCESSING: Parse intent, execute hardware command, reset Vosk.
        """
        p = pyaudio.PyAudio()
        FRAME_RATE = 16000
        CHUNK_SIZE = 2048
        VAD_THRESHOLD = 200  # RMS threshold to trigger listening
        SILENCE_TIMEOUT = 1.5  # Seconds of silence to end listening

        print("=" * 70)
        print("   🚗 DRIVESENSE — IN-VEHICLE AI VOICE ASSISTANT")
        print("=" * 70)
        print(f"  • Hardware Mode: {HARDWARE_MODE.upper()}")
        print(f"  • Wake Word: \"DriveSense\"")
        print("  • Voice Commands:")
        print("     - \"DriveSense\" (wake up)")
        print("     - \"Open the left door\" / \"Close all doors\"")
        print("     - \"Turn on the AC\" / \"Turn off the AC\"")
        print("     - \"Is the AC on?\"")
        print("     - \"What is the cabin temperature?\"")
        print("     - \"Is there an obstacle ahead?\"")
        print("     - \"What is artificial intelligence?\"")
        print("     - \"Stop listening\" (pause) / \"Exit\" (shutdown)")
        print("  • Mic auto-mutes while DriveSense speaks.")
        print("=" * 70 + "\n")

        self.speak("DriveSense is online. Say DriveSense to begin.")

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

            print("\n🔒 [IDLE] Waiting for wake word \"DriveSense\"...")
            
            # State Machine Variables
            audio_state = "WAITING_FOR_SOUND"
            last_loud_time = 0
            accumulated_text = []

            while self.is_running:
                # Mute mic while speaking
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

                # 1. Calculate Volume (RMS)
                audio_np = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_np.astype(float)**2))
                vu_level = min(int(rms / 100), 10)
                vu_bar = "■" * vu_level + " " * (10 - vu_level)

                current_time = time.time()

                # ==========================================
                # STATE 1: WAITING_FOR_SOUND
                # ==========================================
                if audio_state == "WAITING_FOR_SOUND":
                    if rms > VAD_THRESHOLD:
                        audio_state = "LISTENING"
                        last_loud_time = current_time
                        accumulated_text = []
                        # Reset recognizer to clear any old noise buffer
                        rec = KaldiRecognizer(self.vosk_model, FRAME_RATE)
                        rec.SetWords(True)
                        print(f"\r  [{vu_bar}] (Sound Detected - Listening...)      ", end="", flush=True)
                        
                        # Feed this first chunk to Vosk
                        rec.AcceptWaveform(data)
                    else:
                        # Print visual indicator periodically but don't feed to Vosk
                        state_icon = "🔒" if self.assistant_state == AssistantState.IDLE else "🎤"
                        print(f"\r  [{state_icon}] [{vu_bar}] (Quiet)                  ", end="", flush=True)

                # ==========================================
                # STATE 2: LISTENING
                # ==========================================
                elif audio_state == "LISTENING":
                    # Update silence tracker
                    if rms > VAD_THRESHOLD:
                        last_loud_time = current_time

                    # Feed audio to Vosk
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        text_chunk = res.get("text", "").strip()
                        if text_chunk:
                            accumulated_text.append(text_chunk)
                    else:
                        partial_res = json.loads(rec.PartialResult())
                        partial_text = partial_res.get("partial", "").strip()
                        if partial_text:
                            print(f"\r  [{vu_bar}] Hearing: \"{partial_text}\"      ", end="", flush=True)

                    # Check for silence timeout to move to processing
                    if current_time - last_loud_time > SILENCE_TIMEOUT:
                        audio_state = "PROCESSING"

                # ==========================================
                # STATE 3: PROCESSING
                # ==========================================
                if audio_state == "PROCESSING":
                    # Force final result extraction
                    final_res = json.loads(rec.FinalResult())
                    final_chunk = final_res.get("text", "").strip()
                    if final_chunk:
                        accumulated_text.append(final_chunk)

                    full_transcription = " ".join(accumulated_text).strip()

                    # Only process if we actually heard words
                    if full_transcription:
                        if self.assistant_state == AssistantState.IDLE:
                            print(f"\n  [Transcribed in IDLE]: {full_transcription}")
                            if self._is_wake_word(full_transcription):
                                self.assistant_state = AssistantState.ACTIVE
                                print("\n✅ [WAKE WORD DETECTED]")
                                self.speak("Yes?")
                                print("\n🎤 [ACTIVE] Listening for your command...")
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
                                print("\n🎤 [ACTIVE] Speak your next command...")

                    # Reset and return to waiting
                    audio_state = "WAITING_FOR_SOUND"
                    accumulated_text = []

            stream.stop_stream()
            stream.close()

        except KeyboardInterrupt:
            print("\nDriveSense stopping...")
            self.speak("DriveSense shutting down. Goodbye.")
        finally:
            # Cleanup GPIO if on Raspberry Pi
            gpio = self.hardware.get("_gpio")
            if gpio:
                gpio.cleanup()
            p.terminate()


# ==============================================================================
# 10. MAIN ENTRY POINT
# ==============================================================================

def main():
    assistant = DriveSenseVoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()

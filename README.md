# 🚗 DriveSense — AI-Driven In-Vehicle Digital Voice Assistant

**DriveSense** is a realistic, AI-driven, hands-free in-vehicle digital voice assistant designed for automotive environments. It features a **decoupled architecture** separating natural language intent understanding from hardware sensor/actuator execution, with a **hardware abstraction layer** enabling seamless migration from laptop development to **Raspberry Pi** deployment.

---

## 🏗️ System Architecture

```
                       [ Driver Spoken Voice ]
                                  │
                                  ▼
                  [ Offline STT: Vosk (16kHz) ]
                                  │
                                  ▼
    ┌──────────────────────────────────────────────────────────────┐
    │            DriveSense AI Intelligence Layer                  │
    │    13-Intent Classifier + Parameter Extraction               │
    │    Outputs strict JSON: { "intent": "...", "target": "..." } │
    └─────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                  Command Dispatcher                          │
    │       Routes intent → correct hardware controller            │
    └─────────────────────────┬────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌────────────────┐ ┌─────────────┐ ┌─────────────┐
    │ DoorController │ │  TempSensor │ │  ACController│  ...
    │  (Simulated /  │ │ (Simulated /│ │ (Simulated / │
    │   Raspberry Pi)│ │  Raspberry  │ │  Raspberry   │
    └────────────────┘ │  Pi GPIO)   │ │  Pi GPIO)    │
                       └─────────────┘ └─────────────┘
                              │
                              ▼
               [ Two-Way Audio: pyttsx3 TTS ]
                              │
                              ▼
                      [ Vehicle Speaker ]
```

---

## 🎯 Supported Intents (13 Total)

| Intent | Description | Example Voice Commands |
| :--- | :--- | :--- |
| `GREETING` | Greet DriveSense | *"Hello"*, *"Hi DriveSense"* |
| `GET_TEMPERATURE` | Cabin temperature inquiry | *"What is the temperature?"*, *"How hot is the cabin?"* |
| `GET_DISTANCE` | Obstacle distance measurement | *"Is there an obstacle ahead?"*, *"How far is the object?"* |
| `OPEN_DOOR` | Open specific door(s) | *"Open the left door"*, *"Open all doors"* |
| `CLOSE_DOOR` | Close specific door(s) | *"Close the right door"*, *"Close all doors"* |
| `TURN_AC_ON` | Activate air conditioning | *"Turn on the AC"*, *"Switch on the cooling"* |
| `TURN_AC_OFF` | Deactivate air conditioning | *"Turn off the AC"*, *"Stop the AC"* |
| `GET_AC_STATUS` | Query AC state | *"Is the AC on?"*, *"Check AC status"* |
| `GET_SENSOR_STATUS` | Query IR sensor | *"Check the IR sensor"*, *"Is there an obstacle detected?"* |
| `GENERAL_QUERY` | General knowledge question | *"What is machine learning?"*, *"How does GPS work?"* |
| `STOP_LISTENING` | Pause — return to wake word | *"Stop listening"*, *"Go to sleep"* |
| `EXIT` | Complete shutdown | *"Exit"*, *"Goodbye"* |
| `UNSUPPORTED` | Feature not implemented | *"Play music"*, *"Navigate to Chennai"* |

---

## 🔧 Hardware Abstraction Layer

DriveSense uses a **configuration-driven hardware abstraction** so the same AI/voice code runs on both platforms:

```python
# In drivesense.py — change this one variable:
HARDWARE_MODE = "simulation"     # Laptop development
HARDWARE_MODE = "raspberry_pi"   # Raspberry Pi deployment
```

| Abstraction | Simulated (Laptop) | Raspberry Pi (GPIO) |
| :--- | :--- | :--- |
| `TemperatureSensor` | Random 24–30°C | DHT22 on GPIO 4 |
| `DistanceSensor` | Random 15–80cm | HC-SR04 on GPIO 23/24 |
| `IRSensor` | Based on distance threshold | IR sensor on GPIO 17 |
| `DoorController` | Updates VehicleState dict | Servo/motor on GPIO 5,6,12,13,19,26 |
| `ACController` | Updates VehicleState flag | Relay on GPIO 27 |

---

## 🗂️ Project Structure

```
digital-voice-assistant-in-car/
├── drivesense.py               # Complete voice assistant application
├── drivesense_dataset.csv      # 100+ labeled samples for 13 intents
├── Driver_Assistant.ipynb      # ML model training & evaluation notebook
├── requirements.txt            # Python dependencies
├── vosk-model-small-en-in-0.4/ # Offline speech recognition model
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
git clone https://github.com/AMITJHA-hub/Vehicle_AI_Voice_Assistant.git
cd Vehicle_AI_Voice_Assistant

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

### 2. Run DriveSense

```bash
python drivesense.py
```

### 3. Voice Interaction

1. Wait for `[IDLE] Waiting for wake word "DriveSense"...`
2. Say **"DriveSense"** → assistant responds **"Yes?"**
3. Speak commands naturally:
   - *"Open the left door"*
   - *"What's the temperature?"*
   - *"Turn on the AC"*
   - *"Is the AC on?"*
   - *"Is there an obstacle ahead?"*
   - *"Close all doors"*
4. Say **"Stop listening"** → returns to wake word mode
5. Say **"DriveSense"** again → resumes
6. Say **"Exit"** → shuts down

---

## 🍓 Raspberry Pi Deployment

### Hardware Connections

| Sensor / Actuator | RPi Pin | GPIO |
| :--- | :--- | :--- |
| DHT22 Temperature | Pin 7 | GPIO 4 |
| HC-SR04 Trigger | Pin 16 | GPIO 23 |
| HC-SR04 Echo (via divider) | Pin 18 | GPIO 24 |
| IR Obstacle Sensor | Pin 11 | GPIO 17 |
| AC Relay | Pin 13 | GPIO 27 |
| Door Servo LEFT | Pin 29 | GPIO 5 |
| Door Servo RIGHT | Pin 31 | GPIO 6 |
| Door Servo FRONT_LEFT | Pin 32 | GPIO 12 |
| Door Servo FRONT_RIGHT | Pin 33 | GPIO 13 |
| Door Servo REAR_LEFT | Pin 35 | GPIO 19 |
| Door Servo REAR_RIGHT | Pin 37 | GPIO 26 |

### Migration Steps

1. Copy project to Raspberry Pi
2. Install dependencies: `pip install -r requirements.txt`
3. Install Pi-specific libraries: `pip install RPi.GPIO Adafruit_DHT`
4. Change `HARDWARE_MODE = "raspberry_pi"` in `drivesense.py`
5. Connect USB microphone and speaker
6. Run: `python drivesense.py`

The voice/AI/dispatcher code remains **identical** — only the hardware implementations change.

---

## 📄 License
MIT License

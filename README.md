# 🚗 DriveSense: AI-Driven In-Vehicle Digital Voice Assistant

**DriveSense** is an AI-driven, hands-free in-vehicle digital voice assistant designed for automotive environments. It features a decoupled architecture separating natural language intent understanding from hardware sensor execution, making it lightweight, modular, and ready for deployment from development laptops to a **Raspberry Pi** connected to physical vehicle and environmental sensors.

---

## 🏗️ System Architecture

```
                       [ Driver Spoken Voice ]
                                  │
                                  ▼
                [ Offline Speech-to-Text: Vosk ]
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              DriveSense AI Intelligence Layer               │
    │        Classifies Intent & Emits Valid Strict JSON          │
    └─────────────────────────────┬───────────────────────────────┘
                                  │  { "intent": "...", "response": "..." }
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │             Hardware & Sensor Command Dispatcher            │
    │   - Reads DHT Temperature Sensor (GPIO 4 / Simulation)      │
    │   - Reads HC-SR04 Ultrasonic Distance (GPIO 23/24 / Sim)    │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
              [ Two-Way Audio Output: pyttsx3 TTS ]
                                  │
                                  ▼
                          [ Vehicle Speaker ]
```

---

## 🎯 Supported Functions & Intent Classification

DriveSense classifies every request into exactly ONE of the 6 official intents:

| Intent | Description | Example Queries | Action / Execution |
| :--- | :--- | :--- | :--- |
| **`GREETING`** | User greets DriveSense | *"Hello"*, *"Hi DriveSense"* | Responds with conversational greeting |
| **`GET_TEMPERATURE`** | Cabin temperature inquiry | *"What is the temperature?"*, *"Cabin temp"* | Reads DHT temperature sensor (Celsius) |
| **`GET_DISTANCE`** | Front obstacle / collision monitoring | *"Is there an obstacle?"*, *"How far is object?"* | Reads HC-SR04 ultrasonic sensor (cm) |
| **`GENERAL_QUERY`** | General automotive / tech question | *"What is machine learning?"*, *"How does GPS work?"* | Delivers concise, safe spoken answer |
| **`EXIT`** | Stop / shutdown assistant | *"Stop"*, *"Exit"*, *"Goodbye"* | Acknowledges and shuts down safely |
| **`UNSUPPORTED`** | Unimplemented vehicle features | *"Turn on AC"*, *"Call my friend"*, *"Navigate"* | Informs driver feature is not available |

> **Safety Rule:** DriveSense never fabricates sensor readings. The AI layer outputs structured JSON, and the application dispatcher queries the physical sensors.

---

## 📂 Project Structure

```
digital-voice-assistant-in-car/
│
├── drivesense.py               # Main 2-way continuous voice assistant application
├── drivesense_dataset.csv      # Labeled training dataset for all 6 DriveSense intents
├── Driver_Assistant.ipynb      # Interactive Jupyter notebook for ML model training & evaluation
├── requirements.txt            # Python dependencies
├── vosk-model-small-en-in-0.4/ # Offline speech recognition acoustic & language model
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/AMITJHA-hub/Vehicle_AI_Voice_Assistant.git
cd Vehicle_AI_Voice_Assistant

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Running DriveSense Voice Assistant

Start the hands-free, two-way voice assistant:
```bash
python drivesense.py
```

- **Speak directly into your microphone.**
- DriveSense listens in real time, auto-mutes the mic when responding, and speaks the sensor readings through your speaker.

---

## 🍓 Raspberry Pi Deployment (Physical Hardware Sensors)

When deploying to a Raspberry Pi connected to physical sensors:

1. **Hardware Pin Connections:**
   - **DHT22 Temperature Sensor:** VCC $\to$ 3.3V/5V, GND $\to$ GND, Data $\to$ **GPIO 4 (Pin 7)**
   - **HC-SR04 Ultrasonic Distance Sensor:**
     - Trig $\to$ **GPIO 23 (Pin 16)**
     - Echo $\to$ Voltage divider (1kΩ / 2kΩ) $\to$ **GPIO 24 (Pin 18)**

2. **Run on Raspberry Pi:**
   ```bash
   python drivesense.py
   ```
   The hardware dispatcher automatically initializes `RPi.GPIO` and switches from Laptop Simulation Mode to live physical sensor reading!

---

## 📄 License
MIT License

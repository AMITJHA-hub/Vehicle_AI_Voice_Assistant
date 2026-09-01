# 🚗 DriveSense — AI-Driven In-Vehicle Digital Voice Assistant

**DriveSense** is a realistic, AI-driven, hands-free in-vehicle digital voice assistant designed for automotive environments. It features a **decoupled architecture** separating natural language intent understanding from hardware sensor/actuator execution, with a **hardware abstraction layer** enabling seamless migration from laptop development to **Raspberry Pi** deployment.

---

## 🏗️ System Architecture

```
                       [ Driver Spoken Voice ]
                                  │
                                  ▼
             [ Online STT: Google SpeechRecognition ]
                                  │
                                  ▼
    ┌──────────────────────────────────────────────────────────────┐
    │            DriveSense AI Intelligence Layer                  │
    │    Google Gemini 3.6 Flash LLM API                           │
    │    Outputs strict JSON: { "intent": "...", "target": "..." } │
    └─────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                  Command Dispatcher                          │
    │       Routes intent → Simulated VehicleController            │
    └─────────────────────────┬────────────────────────────────────┘
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



## 🗂️ Project Structure

```
digital-voice-assistant-in-car/
├── drivesense.py               # Complete voice assistant application
├── api_key.txt                 # Local file for Google Gemini API key
├── requirements.txt            # Python dependencies
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

### 2. Configure API Key
Create a file named `api_key.txt` in the root folder and paste your Google Gemini API key inside it.
```bash
echo "YOUR_GEMINI_API_KEY" > api_key.txt
```

### 3. Run DriveSense

```bash
python drivesense.py
```

### 4. Voice Interaction

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



## 📄 License
MIT License

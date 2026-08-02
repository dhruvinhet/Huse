# Setup and Running Guide

This guide provides the exact steps and commands required to set up, configure, and run the **Whiteboard AI Video Generator** project.

---

## Prerequisites

Before starting, ensure you have the following installed and available on your system:
* **Python 3.12**
* **Git**
* **FFmpeg** and **FFprobe** (must be on your system `PATH`, or configured via path in `.env`)
* **Gemini API Key** (from Google AI Studio), or an **NVIDIA API Key**

---

## Step-by-Step Setup Instructions

All commands below should be executed in your terminal (such as PowerShell or Command Prompt) targeting this directory.

### Step 1: Create a Python Virtual Environment
Initialize a clean Python virtual environment named `.venv`:
```powershell
python -m venv .venv
```

### Step 2: Activate the Virtual Environment
Activate the environment to ensure Python references are pointing to your localized `.venv`:

* **For Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
* **For Windows (Command Prompt):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
* **For Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
Upgrade pip and install the required libraries:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Configure the Environment Variables
1. Copy the template `.env.example` file to create a `.env` file:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open the newly created `.env` file and replace the `GEMINI_API_KEY` placeholder with your key:
   ```dotenv
   AI_PROVIDER=gemini
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_MODEL=gemini-3.5-flash
   NVIDIA_API_KEY=
   NVIDIA_MODEL=meta/llama-3.1-70b-instruct
   NVIDIA_VISION_MODEL=google/gemma-3n-e4b-it
   NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
   NVIDIA_MAX_TOKENS=16384
   OUTPUT_DIR=outputs
   TEMP_DIR=temp
   LOG_LEVEL=INFO
   FFMPEG_PATH=
   DEBUG_ARTIFACTS=true
   DEBUG_DIR=outputs/debug
   PIPELINE_VERSION=v2
   V2_MAX_REPAIR_ATTEMPTS=2
   V2_ENABLE_MULTIMODAL=false
   ```
   Set `AI_PROVIDER=gemini` for Gemini or `AI_PROVIDER=nvidia` (the spelling
   `nvidea` is also accepted) for NVIDIA. With NVIDIA selected, fill in
   `NVIDIA_API_KEY` and choose the model with `NVIDIA_MODEL`.
   *   **GEMINI_MODEL**: By default, this is set to `gemini-3.5-flash`. If you encounter high demand error (503), you can switch it to another model like `gemini-3.6-flash` or `gemini-3.5-flash-lite`.
   *   **FFMPEG_PATH**: If FFmpeg/FFprobe are not configured in your system `PATH`, set `FFMPEG_PATH` to the absolute folder path containing `ffmpeg.exe` (e.g., `FFMPEG_PATH=C:/ffmpeg/bin`).*

---

## Running the Project

### Run the Interactive Demo Pipeline
To start generating whiteboard educational videos:
```powershell
python run_demo.py
```
Upon running, the script will prompt you:
```text
Enter Topic: Explain the encoder-decoder concept
```
Type your topic and press `Enter`. The pipeline will execute all stages and generate:
* **Final Video**: `outputs/final_video.mp4`
* **Narrated Audio**: `outputs/audio/narration.mp3`
* **Debug Checkpoints**: `outputs/debug/runs/<run_id>/`

---

## Verifying with Tests

Ensure everything is configured and operating correctly by running the suite of unit and integration tests (mocks Gemini/TTS APIs by default):
```powershell
python -m pytest -q
```
You should see the complete test suite pass. The exact count may change as tests
are added or updated.

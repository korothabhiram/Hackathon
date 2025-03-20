# Live Transcription with Twitter Integration

This Python script captures live audio from your microphone, transcribes it in real time using the `faster-whisper` model, and tweets the transcription automatically using the Twitter API.

## Prerequisites

- Python 3.9.14 (recommended to use `pyenv`)
- Twitter Developer Account & API Keys
- Use pip install to download required dependencies: `tweepy`, `sounddevice`, `numpy`, `scipy`, `faster-whisper`

## Installation

### 1. Install `pyenv` and Set Python Version

To ensure compatibility, install and set Python 3.9.14 using `pyenv`:

```bash
# Install pyenv (Mac/Linux)
curl https://pyenv.run | bash

# Restart shell and install Python 3.9.14
pyenv install 3.9.14
pyenv global 3.9.14
```

Verify the installation:
```bash
python --version
```
Expected output:
```
Python 3.9.14
```

### 2. Clone the Repository

```bash
git clone git@github.com:korothabhiram/Hackathon.git
cd Hackathon
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r <"package required to import">
```



### 5. Run the Script

```bash
python app.py
```

## How It Works
- The script records audio from your default microphone.
- It transcribes speech using `faster-whisper`.
- The transcribed text is printed to the console and posted to Twitter.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'tweepy'`**
  - Ensure you have activated the virtual environment using `source venv/bin/activate`.
  - Run `pip install -r tweepy` again.

## License
MIT License



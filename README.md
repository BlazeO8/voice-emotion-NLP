# Voice-Based Emotion Detection using NLP

Speak (or type) a sentence; the app converts speech to text, cleans it, extracts TF-IDF features and predicts
an emotion with a Logistic Regression text classifier. Voice is only the *input*; classification is done on the
recognized **text** (NLP).

## Features
- Voice input (browser microphone via `st.audio_input`; optional system mic via PyAudio)
- Speech-to-text (SpeechRecognition, Google Web Speech API - needs internet)
- NLP preprocessing (lowercase, punctuation removal, tokenisation, stop-words, lemmatisation; negations kept)
- TF-IDF (unigrams + bigrams) + Logistic Regression in one sklearn pipeline
- Emotion prediction with **Model Confidence** and per-class probabilities
- Text-input fallback, editable recognized text, session prediction history
- Friendly error handling (no speech, not understood, no internet, missing model, empty text ...)

## Installation
```bash
pip install -r requirements.txt
```
(Optional, only for the "system microphone" button: `pip install pyaudio`)

## Train Model
```bash
python generate_dataset.py   # optional - dataset is already included
python train_model.py        # creates models/emotion_pipeline.pkl + results/
```

## Run App
```bash
streamlit run app.py
```

## Supported Emotions
Happy, Sad, Angry, Neutral

## Important notes
- `data/emotion_dataset.csv` is a synthetic **educational/demo dataset** (template-based), NOT a real-world dataset.
  Near-perfect test accuracy on it is expected and does NOT reflect real-world performance.
- Audio is processed in memory for speech recognition and is not saved by this application. The audio is sent
  to Google's free speech API for transcription.
- Predictions are statistical text patterns, not a measurement of anyone's psychological state.

## Troubleshooting
- **Model not found** -> run `python train_model.py`.
- **"Speech recognition service unavailable"** -> check internet; use the text box meanwhile.
- **Mic not working in browser** -> allow microphone permission for localhost; use Chrome/Edge/Firefox.
- **`st.audio_input` missing** -> `pip install -U streamlit`.
- **PyAudio install fails** -> it is optional; use the browser recorder instead.
- **NLTK data missing** -> the app downloads it automatically, or falls back to a built-in stop-word list.

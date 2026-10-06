"""Voice-Based Emotion Detection using NLP - Streamlit application."""
import hashlib
import io
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predict import all_probabilities, load_model, predict_emotion

try:
    import speech_recognition as sr
except ImportError:  # handled gracefully in the UI
    sr = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_pipeline.pkl"
DATASET_PATH = BASE_DIR / "emotion_dataset.csv"

st.set_page_config(page_title="Voice-Based Emotion Detection", page_icon="🎙️", layout="centered")

EMOJI = {"happy": "😊", "sad": "😢", "angry": "😠", "neutral": "😐"}
COLOR = {"happy": "#f5b301", "sad": "#3b82f6", "angry": "#ef4444", "neutral": "#6b7280"}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 820px;}
.title {text-align:center; font-size:2.2rem; font-weight:800; margin-bottom:0;}
.subtitle {text-align:center; font-size:1.15rem; color:#6b7280; margin-top:0;}
.desc {text-align:center; color:#6b7280; margin-bottom:1.5rem;}
.result-card {border-radius:16px; padding:1.6rem; text-align:center; color:white; margin:1rem 0;}
.result-emotion {font-size:2.6rem; font-weight:800; letter-spacing:2px;}
.result-conf {font-size:1.2rem; opacity:.95;}
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("history", [])
st.session_state.setdefault("speech_text", "")
st.session_state.setdefault("last_audio_hash", None)
st.session_state.setdefault("last_result", None)


def ensure_model_ready():
    """Use the repo-root dataset/model that already exists in this project."""
    if MODEL_PATH.exists():
        return

    if not DATASET_PATH.exists():
        st.warning("Training dataset not found. Generating it now...")
        subprocess.run([sys.executable, str(BASE_DIR / "generate_dataset.py")], cwd=str(BASE_DIR), check=True)

    st.warning("Model file missing. Training the model now...")
    subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], cwd=str(BASE_DIR), check=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model still missing after training: {MODEL_PATH}")


@st.cache_resource(show_spinner=False)
def get_model():
    ensure_model_ready()
    return load_model()


try:
    MODEL = get_model()
except FileNotFoundError:
    MODEL = None
except Exception:
    MODEL = False


def transcribe_wav_bytes(wav_bytes):
    """Speech-to-text from in-memory WAV bytes (audio is never written to disk)."""
    if sr is None:
        st.error("The SpeechRecognition package is not installed. Run: pip install SpeechRecognition")
        return None
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if frames.size == 0 or np.sqrt(np.mean(frames.astype(np.float64) ** 2)) < 30:
            st.warning("⚠️ No speech detected. Please try again.")
            return None
    except Exception:
        pass
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("⚠️ Sorry, I couldn't understand the speech.\n\nPlease speak clearly and try again.")
    except sr.RequestError:
        st.warning("⚠️ Speech recognition service is unavailable.\n\nPlease check your internet connection or use text input.")
    except Exception:
        st.warning("⚠️ Could not process the audio. Please try again or use text input.")
    return None


def record_from_system_mic():
    """Optional: record using the computer's microphone (needs PyAudio)."""
    if sr is None:
        st.error("The SpeechRecognition package is not installed.")
        return None
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            st.info("🎤 Listening... speak now.")
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=10)
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.warning("⚠️ No speech detected. Please try again.")
    except sr.UnknownValueError:
        st.warning("⚠️ Sorry, I couldn't understand the speech.\n\nPlease speak clearly and try again.")
    except sr.RequestError:
        st.warning("⚠️ Speech recognition service is unavailable.\n\nPlease check your internet connection or use text input.")
    except (OSError, AttributeError):
        st.error("🎙️ No usable microphone found, or PyAudio is not installed / permission was denied. Use the browser recorder above or the text input.")
    except Exception:
        st.warning("⚠️ Microphone error. Please use text input instead.")
    return None


def analyze(text, source):
    """Run the NLP model on `text` and store the result in session state."""
    if MODEL is None:
        st.error("Model file not found. Run `python train_model.py` first.")
        return
    if MODEL is False:
        st.error("The model file could not be loaded. Re-train it with `python train_model.py`.")
        return
    if not text or not text.strip():
        st.warning("⚠️ Please provide some text (speak or type a sentence) before analyzing.")
        return
    try:
        with st.spinner("Analyzing..."):
            emotion, confidence = predict_emotion(text, MODEL)
            probs = all_probabilities(text, MODEL)
    except ValueError as e:
        st.error(f"Could not classify this input: {e}")
        return
    except Exception:
        st.error("Something went wrong while analyzing the text. Please try again.")
        return
    st.session_state.last_result = {"text": text, "emotion": emotion, "confidence": confidence, "probs": probs}
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"), "Source": source, "Text": text,
        "Emotion": f"{EMOJI[emotion]} {emotion.capitalize()}", "Confidence": f"{confidence:.1f}%"})


with st.sidebar:
    st.markdown("### About the Model")
    st.markdown("**Algorithm:**  \nLogistic Regression\n\n**Feature Extraction:**  \nTF-IDF (1-2 grams)\n\n"
                "**Input:**  \nSpeech / Text\n\n**Classes:**  \n😊 Happy · 😢 Sad · 😠 Angry · 😐 Neutral")
    st.markdown("### How it works")
    st.markdown("1. Voice is converted to text.\n2. Text is cleaned.\n3. TF-IDF extracts features.\n"
                "4. NLP model predicts emotion.\n5. Result is displayed.")
    st.info("Trained on an educational/demo dataset, so results on real speech may vary.")
    st.caption("Audio is used for speech recognition and is not permanently stored by this application.")

st.markdown('<div class="title">🎙��� Voice-Based Emotion Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Using Natural Language Processing</div>', unsafe_allow_html=True)
st.markdown('<div class="desc">Speak naturally and let the system analyze the emotion expressed in your sentence.</div>',
            unsafe_allow_html=True)

if MODEL is None:
    st.error("⚠️ Trained model not found (`emotion_pipeline.pkl`). Run `python train_model.py` first.")

with st.container(border=True):
    st.subheader("🎤 Voice Input")
    if hasattr(st, "audio_input"):
        audio = st.audio_input("Click the microphone to record, then stop to transcribe")
        if audio is not None:
            data = audio.getvalue()
            h = hashlib.md5(data).hexdigest()
            if h != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = h
                with st.spinner("Converting speech to text..."):
                    text = transcribe_wav_bytes(data)
                if text:
                    st.session_state.speech_text = text
    else:
        st.info("Your Streamlit version has no browser recorder. Upgrade (`pip install -U streamlit`) or use the system microphone / text input.")
    with st.expander("Use system microphone instead (requires PyAudio)"):
        if st.button("🎙️ Record with system microphone"):
            text = record_from_system_mic()
            if text:
                st.session_state.speech_text = text

    st.markdown("**📝 Recognized Speech** (you can edit it before analyzing)")
    st.text_area("Recognized speech", key="speech_text", height=90, label_visibility="collapsed",
                 placeholder="Your recognized speech will appear here...")
    if st.button("🔍 Analyze Emotion", type="primary", width="stretch"):
        analyze(st.session_state.speech_text, "Voice/Speech box")

st.markdown("<p style='text-align:center;color:#9ca3af'>— OR —</p>", unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("⌨️ Enter Text Manually")
    manual = st.text_input("Type a sentence", placeholder="I am extremely happy today!", key="manual_text")
    if st.button("🔍 Analyze Text", width="stretch"):
        analyze(manual, "Typed")

res = st.session_state.last_result
if res:
    e = res["emotion"]
    st.markdown("### Predicted Emotion")
    st.markdown(
        f'<div class="result-card" style="background:{COLOR[e]}">'
        f'<div class="result-emotion">{EMOJI[e]} {e.upper()}</div>'
        f'<div class="result-conf">Model Confidence: {res["confidence"]:.1f}%</div></div>',
        unsafe_allow_html=True)
    st.markdown(f"> \"{res['text']}\"")
    st.info(f"The model classified this sentence as **{e.capitalize()}** based on patterns learned from the training dataset. This is a text-based prediction, not a measurement of your actual psychological state.")
    with st.expander("See probability for every emotion"):
        pdf = pd.DataFrame({"Emotion": [k.capitalize() for k in res["probs"]],
                            "Probability (%)": [round(v, 1) for v in res["probs"].values()]}).set_index("Emotion")
        st.bar_chart(pdf)

st.markdown("### 🕒 Prediction History (this session)")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history[::-1]), width="stretch", hide_index=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()
else:
    st.caption("No predictions yet.")

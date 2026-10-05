"""
Generates the EDUCATIONAL / DEMO emotion dataset (data/emotion_dataset.csv).

NOTE: This is a synthetic, template-based dataset created for a college project
demo. It is NOT a real-world dataset, so accuracy on it is optimistic and does
not represent performance on real human speech.
"""
import itertools
import os
import random
from pathlib import Path

import pandas as pd

random.seed(42)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emotion_dataset.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

OPENERS = ["", "Honestly, ", "Right now ", "Today ", "Well, ", "You know, ", "Lately ", "Right now, "]

HAPPY = [
    "I am so happy today", "I feel wonderful and full of joy", "I love this wonderful experience",
    "This is the best day of my life", "I am extremely happy because I got excellent results",
    "I am thrilled with how things turned out", "I feel delighted and grateful", "That news made me really excited",
    "I am so proud of what we achieved", "Everything feels amazing and I am smiling all day",
    "I am overjoyed to see my friends again", "What a fantastic surprise, I love it",
    "I feel cheerful and full of energy", "I am really pleased with my results",
    "Life is beautiful and I am so glad", "I had a great time and I am very happy",
    "I am so excited about the trip", "This gift made me incredibly happy",
    "I feel lucky and blessed", "We won the match and I am over the moon",
]
SAD = [
    "I feel very sad and disappointed", "I am heartbroken and crying", "I feel terrible and so lonely",
    "I miss my family and feel miserable", "I am depressed and nothing feels good",
    "I failed the exam and I feel hopeless", "My heart feels heavy and I am very upset",
    "I lost my best friend and I am devastated", "I feel empty and sorrowful",
    "It hurts so much, I just want to cry", "I am unhappy and feel completely alone",
    "I feel down and discouraged", "This is so depressing and I feel awful",
    "I am grieving and cannot stop crying", "I feel gloomy and tired of everything",
    "I am sad that the holidays are over", "Nobody called me and I feel abandoned",
    "I regret everything and feel so low", "I am disappointed with my poor results",
    "I feel sorrow and pain today",
]
ANGRY = [
    "I am really angry because you lied to me", "I hate this, it makes me furious",
    "This is unacceptable and I am so mad", "I am extremely angry at their behaviour",
    "I am sick of being ignored and I am annoyed", "You broke my phone and I am furious",
    "I can not stand this nonsense anymore", "This makes my blood boil",
    "I am outraged by the rude service", "Stop shouting at me, I am very irritated",
    "I am fed up with all these excuses", "How dare you speak to me like that",
    "I am so frustrated and angry with this delay", "I despise the way they cheated us",
    "I am livid about this unfair decision", "That driver made me so angry",
    "I am enraged that nobody listened", "I hate being treated this badly",
    "I am angry that you broke your promise", "This is ridiculous and I am really mad",
]
NEUTRAL = [
    "Today is Monday and I have a class at ten o'clock", "The meeting is scheduled for the afternoon",
    "I am going to the market to buy vegetables", "The train arrives at platform number three",
    "I need to submit the assignment by Friday", "The library opens at nine in the morning",
    "I will drink some tea and read a book", "The weather report says it is cloudy",
    "I have a lecture on computer networks", "I walked to the office this morning",
    "The report contains five sections", "I am reading a chapter about history",
    "We will discuss the project plan tomorrow", "The shop is next to the bus stop",
    "I don't know how I feel about this", "I am not sure what to think about it",
    "The package should arrive on Wednesday", "I usually have lunch at one o'clock",
    "The class has thirty students", "I am updating the document right now",
]
SUFFIXES = ["", "", "", " right now", " at the moment", " this week", " these days", " again"]


def build(label, base):
    rows = set()
    for b, o, s in itertools.product(base, OPENERS, SUFFIXES):
        text = (o + (b if (not o or b.startswith('I ')) else b[0].lower() + b[1:]) + s).strip()
        rows.add(text[0].upper() + text[1:] + ("." if random.random() < 0.5 else ""))
    rows = sorted(rows)
    random.shuffle(rows)
    return [(t, label) for t in rows[:250]]


data = []
for lab, base in [("happy", HAPPY), ("sad", SAD), ("angry", ANGRY), ("neutral", NEUTRAL)]:
    data += build(lab, base)
random.shuffle(data)

pd.DataFrame(data, columns=["text", "emotion"]).to_csv(DATA_PATH, index=False)
print(f"Saved {DATA_PATH} with {len(data)} rows")

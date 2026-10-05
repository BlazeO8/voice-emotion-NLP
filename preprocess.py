"""Reusable NLP preprocessing for the Voice-Based Emotion Detection project."""
import re

# Built-in fallback stop-word list (used only if NLTK data is unavailable).
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with", "about", "to", "from",
    "in", "on", "up", "out", "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "this", "that", "these", "those", "so", "as",
    "will", "would", "can", "could", "should", "just", "than", "then", "there", "here", "what", "which",
    "who", "whom", "when", "where", "how", "into", "over", "under", "again", "once", "own", "s", "t",
}

# Words that carry emotional meaning and must NEVER be removed as stop-words
# (negations and intensifiers change the sentiment of a sentence).
KEEP_WORDS = {
    "no", "not", "nor", "never", "very", "too", "most", "more", "all", "only",
    "dont", "doesnt", "didnt", "cant", "cannot", "wont", "isnt", "arent", "wasnt", "nothing", "nobody",
}

_STOPWORDS = None
_LEMMATIZER = None


def _load_resources():
    """Load NLTK stop-words / lemmatizer. Fall back gracefully if data is missing."""
    global _STOPWORDS, _LEMMATIZER
    if _STOPWORDS is not None:
        return
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        try:
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
        lem = WordNetLemmatizer()
        try:
            lem.lemmatize("tests")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            lem.lemmatize("tests")
        _STOPWORDS, _LEMMATIZER = words, lem
    except Exception:
        # NLTK resources missing/offline -> use the built-in fallback
        _STOPWORDS, _LEMMATIZER = set(_FALLBACK_STOPWORDS), None
    # Normalise apostrophes in stop-words (don't -> dont) to match cleaned text
    _STOPWORDS = {w.replace("'", "") for w in _STOPWORDS} - KEEP_WORDS


def preprocess_text(text):
    """Lowercase -> strip punctuation -> tokenise -> remove stop-words -> lemmatise."""
    if not isinstance(text, str):
        return ""
    _load_resources()
    text = text.lower()
    text = text.replace("'", "").replace("\u2019", "")        # don't -> dont
    text = re.sub(r"[^a-z\s]", " ", text)                      # remove punctuation / digits
    text = re.sub(r"\s+", " ", text).strip()                   # normalise whitespace
    tokens = text.split()                                       # tokenise
    tokens = [t for t in tokens if t not in _STOPWORDS]         # remove stop-words
    if _LEMMATIZER is not None:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    print(preprocess_text("I don't know how I feel about this!"))
    print(preprocess_text("I am SO happy today, it's wonderful!!!"))

import re
import subprocess
import sys
import wave
from pathlib import Path

import torch

from device import get_device
from generate import load_model, sample

VOICE_DIR = Path("voices")
PIPER_VOICE = VOICE_DIR / "en_US-lessac-medium.onnx"
OUT_DIR = Path("results/voice")


def trim_to_sentence(text):
    """Cut the story at its last full sentence (the 500-char limit stops mid-word)."""
    ends = [m.end() for m in re.finditer(r'[.!?]"?', text)]
    return text[: ends[-1]] if ends else text


def split_sentences(text):
    """Split into sentences, keeping the punctuation and any closing quote."""
    parts = re.split(r'(?:(?<=[.!?])|(?<=[.!?]"))\s+', text)
    return [p.strip() for p in parts if p.strip()]


# How each mood is read. speed: length_scale (above 1 = slower).
# volume: Piper normalizes to full loudness first, so 1.0 is the max; calm sits
# at 0.8 to leave room for excited. expression: noise_scale (tone variation).
# rhythm: noise_w_scale (timing variation). Both garble above about 1.3-1.5.
MOODS = {
    "calm":    {"speed": 1.0,  "volume": 0.8,  "expression": 0.667, "rhythm": 0.8, "pause": 0.4},
    "excited": {"speed": 0.95, "volume": 1.0,  "expression": 1.2,   "rhythm": 1.2, "pause": 0.45},
    "sad":     {"speed": 1.25, "volume": 0.55, "expression": 0.4,   "rhythm": 0.5, "pause": 0.7},
    "tense":   {"speed": 1.15, "volume": 0.65, "expression": 0.5,   "rhythm": 0.6, "pause": 0.6},
}

MOOD_WORDS = {
    "sad":     {"alas", "sad", "cried", "cry", "sorry", "lost", "missed"},
    "tense":   {"scary", "scared", "afraid", "dark", "angry", "mad", "monster"},
    "excited": {"hooray", "yay", "wow", "happy", "fun", "laughed", "excited"},
}


def detect_mood(sentence):
    """Pick a mood from the words in the sentence, then from '!', else calm."""
    words = set(re.findall(r"[a-z]+", sentence.lower()))
    for mood in ("sad", "tense", "excited"):     # priority order
        if words & MOOD_WORDS[mood]:
            return mood
    if sentence.rstrip('"').endswith("!"):
        return "excited"
    return "calm"


def speak_piper_narrated(sentences, out_path):
    """Read each sentence in its own mood, with a pause after it."""
    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(str(PIPER_VOICE))
    rate = voice.config.sample_rate
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)          # mono
        wav_file.setsampwidth(2)          # 16-bit samples
        wav_file.setframerate(rate)
        for sentence in sentences:
            mood = detect_mood(sentence)
            settings = MOODS[mood]
            config = SynthesisConfig(
                length_scale=settings["speed"],
                volume=settings["volume"],
                noise_scale=settings["expression"],
                noise_w_scale=settings["rhythm"],
            )
            for chunk in voice.synthesize(sentence, syn_config=config):
                wav_file.writeframes(chunk.audio_int16_bytes)
            # Silence = zero-valued samples: 2 bytes each, rate per second
            wav_file.writeframes(b"\x00\x00" * int(rate * settings["pause"]))
            print(f"  {mood:8} {sentence}")


def speak_say(text, out_path):
    # macOS built-in text-to-speech: no install, robotic but clear
    subprocess.run(
        ["say", "-v", "Samantha", "-o", str(out_path), "--data-format=LEI16@22050", text],
        check=True,
    )


def speak_piper(text, out_path):
    # Piper: a small neural text-to-speech model that runs locally
    from piper import PiperVoice

    voice = PiperVoice.load(str(PIPER_VOICE))
    with wave.open(str(out_path), "wb") as wav_file:
        if hasattr(voice, "synthesize_wav"):    # piper-tts 1.3+
            voice.synthesize_wav(text, wav_file)
        else:                                   # older piper-tts
            voice.synthesize(text, wav_file)


ENGINES = {"say": speak_say, "piper": speak_piper}


if __name__ == "__main__":
    RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "transformer_10000_ctx256"
    ENGINE = sys.argv[2] if len(sys.argv) > 2 else "both"
    PROMPT = sys.argv[3] if len(sys.argv) > 3 else "Once upon a time"
    TEMPERATURE = 0.8
    MAX_NEW_CHARS = 500

    torch.manual_seed(42)
    device = get_device()
    model, stoi, itos = load_model(RUN_NAME, device)
    story = trim_to_sentence(sample(model, stoi, itos, PROMPT, TEMPERATURE, MAX_NEW_CHARS, device))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{RUN_NAME}.txt").write_text(story + "\n")
    print(f"{RUN_NAME} | temperature {TEMPERATURE}\n\n{story}\n")

    engines = ENGINES if ENGINE == "both" else {ENGINE: ENGINES[ENGINE]}
    for name, speak in engines.items():
        out_path = OUT_DIR / f"{RUN_NAME}_{name}.wav"
        speak(story, out_path)
        print(f"saved {out_path}")

    out_path = OUT_DIR / f"{RUN_NAME}_piper_narrated.wav"
    speak_piper_narrated(split_sentences(story), out_path)
    print(f"saved {out_path}")

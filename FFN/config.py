"""
Project configuration and constants.
Shared across data preprocessing, model, training, and evaluation.
"""

# --- Note classes (15: A-flat through G-sharp) ---
NOTE_CLASSES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]
NUM_CLASSES = len(NOTE_CLASSES)

# --- Spectrogram / audio (librosa) ---
NORMALIZE_INPUT = True   # per-sample zero mean, unit variance (often improves accuracy)
SR = 22050          # sample rate (Hz)
N_FFT = 2048        # FFT window size
HOP_LENGTH = 512    # samples between frames
N_MELS = 128        # mel bands (optional; for mel spectrogram)
DURATION = 1.0      # seconds per clip (fixed-length input)

INPUT_SIZE = N_MELS * (1 + (int(SR * DURATION) // HOP_LENGTH))

# --- Paths ---
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "output"
VIS_DIR = "visualizations"

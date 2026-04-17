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

# --- Model architecture ---
# Input: flattened spectrogram shape (e.g. n_mels * time_frames)
# These can be computed from audio params: time_frames ≈ (SR * DURATION) // HOP_LENGTH
INPUT_SIZE = N_MELS * (1 + (int(SR * DURATION) // HOP_LENGTH))
HIDDEN_SIZES = [512, 256, 128]  # hidden layer dimensions (Robert: tune layers & nodes)
ACTIVATION = "relu"             # "relu", "tanh", "leaky_relu"
DROPOUT = 0.3

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# --- Paths ---
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "output"
VIS_DIR = "visualizations"

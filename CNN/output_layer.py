"""
Output layer reference for the note-classification.
Responsibility: Evelyn Yaskin — output layer & transcription to notation.
"""

import torch
import torch.nn as nn

# 12 note classes  
NOTE_CLASSES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]
NUM_CLASSES = len(NOTE_CLASSES)  # 12

# Output layer: last hidden size -> 12 logits -> softmax for class probabilities.
class OutputLayer(nn.Module):
    def __init__(self, hidden_size : int, num_classes : int = NUM_CLASSES):
        super().__init__()
        # Initializes the output layer as a linear transformation.
        self.output = nn.Linear(hidden_size, num_classes)

    # Forward pass: applies the linear transformation to the input tensor.
    def forward(self, x):
        # Applies the linear transformation to the input tensor.
        return self.output(x)

def predict_note(logits):
    # Calculates the softmax probabilities of the logits.
    probs = torch.softmax(logits, dim=-1)
    idx = torch.argmax(probs).item()
    return NOTE_CLASSES[idx]
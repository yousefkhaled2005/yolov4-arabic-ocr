"""
CRNN-CTC Cursive Arabic Recognizer
"""
import torch
import torch.nn as nn

class CRNNRecognizer(nn.Module):
    def __init__(self, num_classes=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        return features

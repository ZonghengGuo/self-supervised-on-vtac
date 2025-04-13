import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(
        self, chan_1, chan_2, chan_3, ks1, ks2, ks3, channels, dropout_prob=0.5
    ):
        super(FCN, self).__init__()

        pd1 = (ks1 - 1) // 2
        pd2 = (ks2 - 1) // 2
        pd3 = (ks3 - 1) // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(channels, chan_1, kernel_size=ks1, stride=1, padding=pd1),
            nn.BatchNorm1d(chan_1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(chan_1, chan_2, kernel_size=ks2, stride=1, padding=pd2),
            nn.BatchNorm1d(chan_2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(chan_2, chan_3, kernel_size=ks3, stride=1, padding=pd3),
            nn.BatchNorm1d(chan_3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.AdaptiveMaxPool1d(1),
        )

        self.projector = nn.Sequential(
            nn.Linear(chan_3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1),
        )

    def extract_features(self, x):
        x = self.encoder(x).squeeze(-1)        # shape: (B, chan_3)
        x = x.view(x.size(0), -1)              # flatten
        features = self.projector(x)           # shape: (B, 64)
        return features

    def forward(self, signal, random_s=None):
        s_f = self.extract_features(signal)

        if random_s is not None:
            r_f = self.extract_features(random_s)
            return self.classifier(s_f), s_f, r_f

        return self.classifier(s_f)

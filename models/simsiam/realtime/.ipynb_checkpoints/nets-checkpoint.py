import torch
import torch.nn as nn


# class DownstreamModel(nn.Module):
#     def __init__(self, encoder,
#                  dropout_prob=0.1):
#         super().__init__()

#         # 冻结 encoder
#         # for param in encoder.parameters():
#         #     param.requires_grad = False
#         self.encoder = encoder

#         self.convs = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=51, padding=25),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Dropout(dropout_prob),
#             nn.Conv1d(32, 64, kernel_size=25, padding=12),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(dropout_prob),
#             nn.Conv1d(64, 128, kernel_size=13, padding=6),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(dropout_prob),
#             nn.AdaptiveMaxPool1d(1)
#         )

#         self.flatten = nn.Flatten()
#         self.classifier = nn.Sequential(nn.Dropout(dropout_prob), nn.Linear(64, 1))

#         self.signal_feature = nn.Sequential(
#             nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_prob)
#         )

#     def forward(self, x, random_s=None):
#         _, features = self.encoder(x)
#         print(features.shape, x.shape)
#         features = features.unsqueeze(1)
#         out = self.convs(features).squeeze(-1)
#         out = out.view(-1, out.size(1))
#         s_f = self.signal_feature(out)

#         if random_s is not None:
#             _, features_random = self.encoder(random_s)
#             features_random = features_random.unsqueeze(1)
#             random_s = self.convs(features_random).squeeze(-1)
#             random_s = random_s.view(-1, random_s.size(1))
#             random_s = self.signal_feature(random_s)

#             return self.classifier(s_f), s_f, random_s

#         return self.classifier(s_f)

class FinetuneModel(nn.Module):
    def __init__(self, pretrained_fcn_encoder, num_classes):
        super(FinetuneModel, self).__init__()
        self.encoder = pretrained_fcn_encoder
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, random_s=None):
        # 主输入
        features = self.encoder.extract_features(x)  # 特征 shape: (B, 64)
        logits = self.classifier(features)

        if random_s is not None:
            features_random = self.encoder.extract_features(random_s)
            return logits, features, features_random

        return logits



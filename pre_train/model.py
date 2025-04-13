import torch
import torch.nn as nn
import resnet


class DirectFwd(nn.Module):
    def __init__(self):
        super(DirectFwd, self).__init__()

    def forward(self, x):
        return x


# class SimSiam(nn.Module):
#     """
#     Build a SimSiam model.
#     """

#     def __init__(self, dim=512, pred_dim=256, projector=True, predictor=True, single_source_mode=False,
#                  encoder='ResNet18'):
#         """
#         dim: feature dimension (default: 2048)
#         pred_dim: hidden dimension of the predictor (default: 512)
#         """
#         super(SimSiam, self).__init__()

#         # create the encoder
#         # num_classes is the output fc dimension, zero-initialize last BNs
#         self.encoder = getattr(resnet, encoder, None)(num_classes=dim)
#         # print(f"encoder is {self.encoder}")
#         self.single_source_mode = single_source_mode

#         if projector:
#             # build a 3-layer projector
#             prev_dim = self.encoder.fc.weight.shape[1]

#             self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
#                                             nn.BatchNorm1d(prev_dim),
#                                             nn.ReLU(inplace=True),  # first layer
#                                             nn.Linear(prev_dim, prev_dim, bias=False),
#                                             nn.BatchNorm1d(prev_dim),
#                                             nn.ReLU(inplace=True),  # second layer
#                                             self.encoder.fc,
#                                             nn.BatchNorm1d(dim, affine=False))  # output layer
#             self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

#         if predictor:
#             # build a 2-layer predictor

#             self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
#                                            nn.BatchNorm1d(pred_dim),
#                                            nn.ReLU(inplace=True),  # hidden layer
#                                            nn.Linear(pred_dim, dim))  # output layer
#         else:
#             self.predictor = DirectFwd()

#     def forward(self, x1, x2):
#         """
#         Input:
#             PPG_G: first views of images
#             PPG_B: second views of images
#         Output:
#             p1, p2, z1, z2: predictors and targets of the network
#             See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
#         """

#         if not self.single_source_mode:
#             # compute features for one view
#             z1, features1 = self.encoder(x1)  # NxC
#             z2, features2 = self.encoder(x2)  # NxC

#             p1 = self.predictor(z1)  # NxC
#             p2 = self.predictor(z2)  # NxC

#             return p1, p2, z1.detach(), z2.detach()

# based on FCN
class SimSiam(nn.Module):
    def __init__(self, dim=64, pred_dim=32, predictor=True, single_source_mode=False, encoder=None):
        super(SimSiam, self).__init__()

        # encoder: 应该传入的是一个 FCN 对象（已带 projector）
        self.encoder = encoder
        self.single_source_mode = single_source_mode

        # predictor head
        if predictor:
            self.predictor = nn.Sequential(
                nn.Linear(dim, pred_dim, bias=False),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(inplace=True),
                nn.Linear(pred_dim, dim)
            )
        else:
            self.predictor = DirectFwd()

    def forward(self, x1, x2):
        if not self.single_source_mode:
            # 提取表示 z1 和 z2（不带 grad 的 target）
            z1 = self.encoder.extract_features(x1)
            z2 = self.encoder.extract_features(x2)

            p1 = self.predictor(z1)
            p2 = self.predictor(z2)

            return p1, p2, z1.detach(), z2.detach()


class FCN(nn.Module):
    def __init__(
        self, chan_1=128, chan_2=256, chan_3=128, ks1=51, ks2=25, ks3=13, channels=3, dropout_prob=0.1
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
        x = self.encoder(x).squeeze(-1)
        x = x.view(x.size(0), -1)
        features = self.projector(x)
        return features

    def forward(self, signal, random_s=None):
        s_f = self.extract_features(signal)

        if random_s is not None:
            r_f = self.extract_features(random_s)
            return self.classifier(s_f), s_f, r_f

        return self.classifier(s_f)


# 测试代码
if __name__ == "__main__":
    # 初始化 FCN 编码器
    fcn_encoder = FCN()
    
    # 初始化 SimSiam
    model = SimSiam(
        dim=64,  # FCN projector 输出的维度
        pred_dim=32,
        predictor=True,
        single_source_mode=False,
        encoder=fcn_encoder
    )
    
    # 输入为两个视角的增强数据
    x1, x2 = torch.randn(32, 3, 7500), torch.randn(32, 3, 7500)
    p1, p2, z1, z2 = model(x1, x2)

    print("p1 shape:", p1.shape) # (32, 64)
    print("p2 shape:", p2.shape)
    print("z1 shape:", z1.shape)
    print("z2 shape:", z2.shape)
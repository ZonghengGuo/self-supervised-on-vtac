import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F


def convert_resnet2d_to_1d(model_2d, in_channels=2):
    model_1d = model_2d

    def conv2d_to_conv1d(conv2d):
        return nn.Conv1d(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size[0],
            stride=conv2d.stride[0],
            padding=conv2d.padding[0],
            dilation=conv2d.dilation[0],
            groups=conv2d.groups,
            bias=(conv2d.bias is not None)
        )

    def bn2d_to_bn1d(bn2d):
        return nn.BatchNorm1d(bn2d.num_features)

    def adapt_avgpool2d_to_1d(pool2d):
        return nn.AdaptiveAvgPool1d(output_size=pool2d.output_size[0])

    for name, module in model_1d.named_modules():
        if isinstance(module, nn.Conv2d):
            new_module = conv2d_to_conv1d(module)
        elif isinstance(module, nn.BatchNorm2d):
            new_module = bn2d_to_bn1d(module)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            new_module = adapt_avgpool2d_to_1d(module)
        else:
            new_module = None

        if new_module is not None:
            parent_module = model_1d
            subnames = name.split(".")
            for subname in subnames[:-1]:
                parent_module = getattr(parent_module, subname)
            setattr(parent_module, subnames[-1], new_module)

    # 修改第一层的输入通道数
    model_1d.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model_1d


class DirectFwd(nn.Module):
    def __init__(self):
        super(DirectFwd, self).__init__()

    def forward(self, x):
        return x


class SimSiam(nn.Module):
    def __init__(self, dim=64, pred_dim=32, predictor=True, encoder=None):
        super(SimSiam, self).__init__()

        # encoder: 应该传入的是一个 FCN 对象（已带 projector）
        self.encoder = encoder

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
        # 提取表示 z1 和 z2（不带 grad 的 target）
        z1 = self.encoder.extract_features(x1)
        z2 = self.encoder.extract_features(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()


class FCN(nn.Module):
    def __init__(
            self, chan_1=128, chan_2=256, chan_3=128, ks1=51, ks2=25, ks3=13, channels=2, dropout_prob=0.1
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1, dropout_prob=0.1):
        super(ResNet, self).__init__()
        self.in_planes = 48

        self.conv0 = nn.Conv1d(2, 48, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(48)
        self.stage0 = self._make_layer(block, 48, num_blocks[0], stride=1)
        self.stage1 = self._make_layer(block, 96, num_blocks[1], stride=2)
        self.stage2 = self._make_layer(block, 192, num_blocks[2], stride=2)
        self.stage3 = self._make_layer(block, 384, num_blocks[3], stride=2)

        self.projector = nn.Sequential(
            nn.Linear(1536, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.fc = nn.Linear(384 * block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def extract_features(self, x):
        out = self.conv0(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.stage0(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        x = F.adaptive_avg_pool1d(out, 1)  # [B, 1536, 1]
        x = x.view(x.size(0), -1)  # [B, 1536]
        features = self.projector(x)

        return features

    def forward(self, signal, random_s=None):
        s_f = self.extract_features(signal)

        if random_s is not None:
            r_f = self.extract_features(random_s)
            return self.classifier(s_f), s_f, r_f

        return self.classifier(s_f)

def ResNet50(num_classes=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=2, seq_len=7500, embed_dim=128, num_heads=8, num_layers=4, output_dim=64, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Linear layer to project input to embedding dimension
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear layer to project Transformer output to desired output dimension
        self.output_projection = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim, seq_len) -> (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)

        # Project input to embedding space
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer expects (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        # Pass through Transformer encoder
        x = self.transformer_encoder(x)

        # Take the mean across the sequence dimension
        x = x.mean(dim=0)

        # Project to output dimension
        x = self.output_projection(x)

        return x


# 测试代码
if __name__ == "__main__":
    # 初始化 FCN 编码器
    # fcn_encoder = FCN()
    # resnet_encoder = ResNet50()

    # # 初始化 SimSiam
    # model_fcn = SimSiam(
    #     dim=64,  # FCN projector 输出的维度
    #     pred_dim=32,
    #     predictor=True,
    #     encoder=fcn_encoder
    # )

    # model_resnet = SimSiam(
    #     dim=64,  # FCN projector 输出的维度
    #     pred_dim=32,
    #     predictor=True,
    #     encoder=resnet_encoder
    # )

    # # 输入为两个视角的增强数据
    # x1, x2 = torch.randn(32, 2, 7500), torch.randn(32, 2, 7500)
    # p1, p2, z1, z2 = model_fcn(x1, x2)


    # print("p1 shape:", p1.shape)  # (32, 64)
    # print("p2 shape:", p2.shape)
    # print("z1 shape:", z1.shape)
    # print("z2 shape:", z2.shape)

    # x1, x2 = torch.randn(32, 2, 7500), torch.randn(32, 2, 7500)
    # p1, p2, z1, z2 = model_resnet(x1, x2)

    # print("p1 shape:", p1.shape)  # (32, 64)
    # print("p2 shape:", p2.shape)
    # print("z1 shape:", z1.shape)
    # print("z2 shape:", z2.shape)
    # Initialize Transformer encoder
    transformer_encoder = TransformerEncoder()

    # Input data
    x1, x2 = torch.randn(16, 2, 7500), torch.randn(16, 2, 7500)

    # Forward pass
    p1 = transformer_encoder(x1)
    p2 = transformer_encoder(x2)

    print("p1 shape:", p1.shape)
    print("p2 shape:", p2.shape)
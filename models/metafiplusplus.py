import torchvision
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from models.channel_trans import ChannelTransformer


class MetaFiPlusPlusNet(nn.Module):
    def __init__(self, num_keypoints=14, num_coor=3, num_person=1,
                 dataset='person-in-wifi-3d', pretrained=False):
        super().__init__()
        self.num_keypoints, self.num_coor = num_keypoints, num_coor
        self.num_person, self.dataset = num_person, dataset
        self.diff = num_keypoints * num_person - 17
        rn = torchvision.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        self.encoder_conv1_p1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.encoder_bn1_p1 = rn.bn1
        self.encoder_relu_p1 = rn.relu
        self.encoder_layer1_p1 = rn.layer1
        self.encoder_layer2_p1 = rn.layer2
        self.encoder_layer3_p1 = rn.layer3
        self.encoder_layer4_p1 = rn.layer4
        self.tf = ChannelTransformer([num_keypoints * num_person, 12], 512, 1, 3,
                                     num_keypoints * num_person)
        self.decode = nn.Sequential(
            nn.Conv2d(512, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_coor, 1, 1, 0, bias=False), nn.BatchNorm2d(num_coor), nn.ReLU(inplace=True))
        self.bn1 = nn.BatchNorm1d(num_coor)
        self.bn2 = nn.BatchNorm2d(512)

        # Pre-build Resize once — creating it inside forward() every call is very slow
        H = 136 + 8 * self.diff    # 136 for MMFI (17kp), 112 for PWIF3D (14kp)
        self._resize = Resize([H, 32])

        # Pool for final spatial collapse — width=12 comes from ResNet output at W=32
        # Verified: Resize→(H,32) → ResNet → H'×2 per antenna → cat 3 → H'×6
        # ChannelTransformer expects (B, 512, num_kp, 12), so pool to width=12
        self._final_pool = nn.AvgPool2d((1, 12), stride=(1, 1))

    def _enc(self, xi):
        xi = self.encoder_relu_p1(self.encoder_bn1_p1(self.encoder_conv1_p1(xi)))
        xi = self.encoder_layer1_p1(xi); xi = self.encoder_layer2_p1(xi)
        xi = self.encoder_layer3_p1(xi); xi = self.encoder_layer4_p1(xi)
        return xi

    def forward(self, x):
        # x: (B, 3, H_csi, W_csi)  e.g. (B, 3, 114, 10) for MMFI
        b = x.shape[0]
        outs = []
        for c in range(3):
            # Take one antenna slice: (B, 1, H_csi, W_csi)
            xc = x[:, c:c+1, :, :]
            # Resize to (B, 1, H_target, 32) — H_target=136 for MMFI, 112 for PWIF3D
            xc = self._resize(xc)
            outs.append(self._enc(xc))          # each: (B, 512, H', W')
        # Concatenate along width: (B, 512, H', 3*W')
        x = self.bn2(torch.cat(outs, dim=3))
        x, _ = self.tf(x)                       # (B, 512, num_kp, 12) after reconstruct
        fea = x.mean(3).mean(2)                 # (B, 512)
        x = self.decode(x)                      # (B, num_coor, num_kp, 12)
        x = self._final_pool(x).squeeze(dim=3)  # (B, num_coor, num_kp)
        x = self.bn1(x)
        x = torch.transpose(x, 1, 2).view(b, self.num_person, self.num_keypoints, self.num_coor)
        return x, fea


def metafiplusplus_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

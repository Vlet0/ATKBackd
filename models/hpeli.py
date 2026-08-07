import torch
import torch.nn as nn
from models.sk_network import SKUnit


class HPELiNet(nn.Module):
    def __init__(self, num_keypoints=14, num_coor=3, subcarrier_num=180,
                 num_person=1, dataset='person-in-wifi-3d'):
        super().__init__()
        self.num_keypoints, self.num_coor = num_keypoints, num_coor
        self.num_person, self.dataset = num_person, dataset
        num_lay = 64

        # For Person-in-WiFi-3D: subcarrier_num=180, packets=20
        # For MMFI:              subcarrier_num=114, packets=10
        # SKUnit dim1 = subcarrier_num (height), dim2 = n_pkt (width, unused in pool)
        self.skunit1 = SKUnit(3, num_lay, num_lay, dim1=subcarrier_num, dim2=10,
                              pool_dim='freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.skunit2 = SKUnit(num_lay, num_lay * 2, num_lay * 2,
                              dim1=subcarrier_num // 2, dim2=8,
                              pool_dim='freq-chan', M=1, G=64, r=4, stride=1, L=32)

        # Compute regression head input size dynamically using a dummy forward pass
        # Input: (1, 3, subcarrier_num, n_pkt) where n_pkt is inferred from subcarrier_num
        # Person-in-WiFi-3D: 180 sub → 20 pkt  |  MMFI: 114 sub → 10 pkt
        n_pkt = 10 if subcarrier_num == 114 else 20
        with torch.no_grad():
            dummy = torch.zeros(1, 3, subcarrier_num, n_pkt)
            pool = nn.AvgPool2d((2, 2))
            h = pool(self.skunit1(dummy))
            h = pool(self.skunit2(h))
            # regression conv layers
            h = nn.ReLU()(nn.Conv2d(128, 64, (3, 1), (2, 1), 0)(h))
            h = nn.ReLU()(nn.Conv2d(64, 32, (3, 1), (2, 1), 0)(h))
            h = nn.ReLU()(nn.Conv2d(32, 16, (3, 1), (1, 1), 0)(h))
            flat_size = h.numel()

        self.regression = nn.Sequential(
            nn.Conv2d(128, 64, (3, 1), (2, 1), 0), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 1), (2, 1), 0), nn.ReLU(),
            nn.Conv2d(32, 16, (3, 1), (1, 1), 0), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, num_keypoints * num_coor * num_person))

        # Pre-build pool — avoid creating nn.AvgPool2d inside forward() every call
        self._pool = nn.AvgPool2d((2, 2))

    def forward(self, x):
        b = x.shape[0]
        x = self._pool(self.skunit1(x))
        out1 = self._pool(self.skunit2(x))
        fea = out1.mean(3).mean(2)
        x = self.regression(out1)
        x = x.reshape(b, self.num_person, self.num_keypoints, self.num_coor)
        return x, fea


def hpeli_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

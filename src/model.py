import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Upsample from in_ch to out_ch
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # After upsampling, we concat with skip (out_ch from encoder)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1 = decoder features, x2 = skip connection
        x1 = self.up(x1)

        # pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX != 0 or diffY != 0:
            x1 = nn.functional.pad(x1, [diffX//2, diffX-diffX//2,
                                        diffY//2, diffY-diffY//2])

        x = torch.cat([x2, x1], dim=1)   # concat along channel axis
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3]*2)

        self.up3 = Up(features[3]*2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up1 = Up(features[2], features[1])
        self.final_up = nn.Conv2d(features[1], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)          # 64
        x2 = self.down1(x1)       # 128
        x3 = self.down2(x2)       # 256
        x4 = self.down3(x3)       # 512
        b = self.bottleneck(x4)   # 1024
        u3 = self.up3(b, x4)
        u2 = self.up2(u3, x3)
        u1 = self.up1(u2, x2)
        out = self.final_up(u1)
        # output is raw; we'll apply loss directly (regression)
        return out

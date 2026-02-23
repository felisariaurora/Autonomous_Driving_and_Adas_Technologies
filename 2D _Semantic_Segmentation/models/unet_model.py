import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Il mattone base della U-Net.
    Invece di fare una sola convoluzione, ne fa sempre due di fila.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Blocco di discesa (Encoder).
    Prima dimezza l'immagine, poi estrae le feature.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # Dimezza altezza e larghezza (es. 512 -> 256)
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Blocco decoder: upsampling trasponibile + concatenazione skip connection + DoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: feature map dal layer più profondo (da upsampare)
        # x2: skip connection dal corrispondente livello encoder

        # 1. Upsampling
        x1 = self.up(x1)

        # 2. Padding per gestire dimensioni dispari (es. input di taglia dispari)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 3. Concatenazione lungo la dimensione canali (encoder + decoder)
        x = torch.cat([x2, x1], dim=1)

        # 4. DoubleConv sul tensore concatenato
        return self.conv(x)


class OutConv(nn.Module):
    """
    Convoluzione 1x1 finale: mappa i canali dell'ultimo decoder alle n_classes di output.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Assemblaggio finale della U-Net.
    Struttura: Encoder (Giù) -> Bottleneck -> Decoder (Su)
    """
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- ENCODER ---
        # Aumento i canali mentre scendo (64 -> 128 -> 256 -> 512 -> 1024)
        # Riduco lo spazio (H, W) dimezzando ogni volta
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024) # Questo è il fondo (Bottleneck)

        # --- DECODER ---
        # Diminuisco i canali mentre salgo
        # Aumento lo spazio (H, W) raddoppiando ogni volta
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 1. Discesa
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 2. Fondo
        x5 = self.down4(x4)

        # 3. Salita
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 4. Classificazione finale
        logits = self.outc(x)
        return logits
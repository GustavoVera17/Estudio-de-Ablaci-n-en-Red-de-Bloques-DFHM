import torch
import torch.nn as nn

class SpectralHallucinationBlock(nn.Module):
    def __init__(self, in_channels, S=2):
        super(SpectralHallucinationBlock, self).__init__()
        self.compressed_channels = in_channels // S
        self.hallucinated_channels = in_channels - self.compressed_channels
        
        # Compresión extra-ligera (1x1)
        self.compress_conv = nn.Conv2d(in_channels, self.compressed_channels, kernel_size=1)
        
        # Alucinación (Depth-wise)
        self.hallucinate_conv = nn.Conv2d(self.compressed_channels, self.hallucinated_channels, 
                                          kernel_size=3, padding=1, 
                                          groups=self.compressed_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        intrinsic_features = self.relu(self.compress_conv(x))
        ghost_features = self.relu(self.hallucinate_conv(intrinsic_features))
        out = torch.cat([intrinsic_features, ghost_features], dim=1)
        return out

class SpatialContextAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SpatialContextAttentionBlock, self).__init__()
        # Depth-wise convolutions para mantener el peso mínimo
        self.conv_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv_7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        
        self.attention_conv = nn.Conv2d(channels * 3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale_1 = self.conv_3x3(x)
        scale_2 = self.conv_5x5(x)
        scale_3 = self.conv_7x7(x)
        
        concat = torch.cat([scale_1, scale_2, scale_3], dim=1)
        attention_map = self.sigmoid(self.attention_conv(concat))
        return x * attention_map

class DFHM(nn.Module):
    def __init__(self, channels, S=2):
        super(DFHM, self).__init__()
        self.shb = SpectralHallucinationBlock(channels, S)
        self.scab = SpatialContextAttentionBlock(channels)

    def forward(self, x):
        residual = x
        out = self.shb(x)
        out = self.scab(out)
        return out + residual 

class ELWRYMNet(nn.Module):
    # Por defecto iniciamos con 64 canales para la versión ligera
    def __init__(self, in_channels=1, out_channels=31, num_features=64, num_blocks=9, S=2):
        super(ELWRYMNet, self).__init__()
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        blocks = [DFHM(num_features, S) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        out = self.tail(x)
        return out

if __name__ == "__main__":
    dummy_input = torch.randn(1, 1, 48, 48)
    modelo = ELWRYMNet(num_blocks=9)
    params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    peso_mb = (params * 4) / (1024 ** 2)
    print(f"--- ELWRYMNet (9 Bloques) ---")
    print(f"Total Parámetros: {params:,}")
    print(f"Peso del modelo : {peso_mb:.4f} MB")
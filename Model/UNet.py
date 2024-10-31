import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        def CCR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
            )
        
        def CC(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
            )

        # Encoder path
        self.encoder1 = CCR(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder2 = CCR(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.encoder3 = CCR(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.encoder4 = CCR(128, 256)

        # Bottleneck
        self.bottleneck = CCR(256, 256)

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = CCR(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = CCR(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = CC(64, 32)

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder path
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.out(dec1)

        return torch.softmax(out, dim=1), bottleneck

class UNet_mod(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Padding = 1 will make sure that the output dimensions will remain the same and will not be reduced by the kernel function
        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(inplace=True)
            )
        
        # Encoder path
        self.e1 = conv_block(in_channels, 64)
        self.p1 = nn.MaxPool2d(kernel_size=2)

        self.e2 = conv_block(64, 128)
        self.p2 = nn.MaxPool2d(kernel_size=2)

        self.e3 = conv_block(128, 256)
        self.p3 = nn.MaxPool2d(kernel_size=2)

        self.e4 = conv_block(256, 512)
        self.p4 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.b = conv_block(512, 1024)

        # Decoder path
        self.u4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d4 = conv_block(1024, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d3 = conv_block(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d2 = conv_block(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d1 = conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder path
        s1 = self.e1(x)
        p1 = self.p1(s1)

        s2 = self.e2(p1)
        p2 = self.p2(s2)

        s3 = self.e3(p2)
        p3 = self.p3(s3)

        s4 = self.e4(p3)
        p4 = self.p4(s4)

        # Bottleneck
        bottleneck = self.b(p4)

        # Decoder path
        d4 = self.u4(bottleneck)
        d4 = torch.cat((s4, d4), dim=1)
        d4 = self.d4(d4)

        d3 = self.u3(d4)
        d3 = torch.cat((s3, d3), dim=1)
        d3 = self.d3(d3)

        d2 = self.u2(d3)
        d2 = torch.cat((s2, d2), dim=1)
        d2 = self.d2(d2)

        d1 = self.u1(d2)
        d1 = torch.cat((s1, d1), dim=1)
        d1 = self.d1(d1)
        out = self.out(d1)

        return torch.sigmoid(out), bottleneck

if __name__ == "__main__" : 
    model = UNet()
    print(model)
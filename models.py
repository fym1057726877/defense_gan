import torch
import torch.nn as nn

# datasets:MNIST

class MnistCnn(nn.Module):
    def __init__(self, ):
        super(MnistCnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.25)

        )
        self.dense = nn.Sequential(
            nn.Linear(6400, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        # x= input tensor
        output = self.conv(x)
        output = output.view(output.shape[0], -1)
        output = self.dense(output)
        # output is now a tensor of shape (N,10) this is pre-softmax logits of N images
        soft = nn.Softmax(dim=1)
        output = soft(output)
        return output


class Generator(nn.Module):

    def __init__(self, in_channels=1, latent_dim=100, conv_channels=32):
        super(Generator, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, conv_channels * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(conv_channels * 4, conv_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(conv_channels * 2, conv_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(conv_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.block(x)
        return output   # output = (batch_size,1,28,28)



class Discriminator(nn.Module):
    def __init__(self, in_channels=1, conv_channels=32):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(

            nn.Conv2d(in_channels, conv_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_channels * 2, conv_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(conv_channels * 4, in_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.block(x)
        output = output.view(-1, 1)
        return output


def test_dis():
    x = torch.randn((16, 1, 28, 28))
    d = Discriminator(in_channels=1, conv_channels=32)
    out = d(x)
    print(out.shape)


def test_gen():
    x = torch.randn((16, 100, 1, 1))
    g = Generator(in_channels=1, latent_dim=100, conv_channels=32)
    out = g(x)
    print(out.shape)

if __name__ == '__main__':
    test_dis()
    test_gen()















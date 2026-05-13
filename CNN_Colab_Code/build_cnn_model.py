# CNN Arcitechture defintion according to the article


class NCSU_DRCNN(nn.Module):
    def __init__(self):
        super(NCSU_DRCNN, self).__init__()
        # Conv1: 32 filters, 3x3
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) # 200 -> 100
        )
        # Conv2: 16 filters, 3x3
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) # 100 -> 50
        )
        # Conv3: 16 filters, 3x3
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) # 50 -> 25
        )
        # Conv4: 32 filters, 3x3
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) # 25 -> 12
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Binary Classification
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
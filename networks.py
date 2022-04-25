import torch
import torch.nn as nn

class ReconstructionCNN2D(nn.Module):
    """
    CNN for reconstructing 2D data.
    """
    
    def __init__(self, num_input_channels, num_output_channels):
        # Call super constructer
        super(ReconstructionCNN2D, self).__init__()

        # Activation function
        self.relu = nn.ReLU()

        # Series of convolutional layers
        self.conv1 = nn.Conv2d(num_input_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv4 = nn.Conv2d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv5 = nn.Conv2d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv6 = nn.Conv2d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv7 = nn.Conv2d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv8 = nn.Conv2d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv9 = nn.Conv2d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv10 = nn.Conv2d(48 * num_output_channels, num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)

        # Initialize weights in the network: all weights following Kaiming normal and all biases are 0
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu', a=0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        return self.conv10(x)  

class ReconstructionCNN3D(nn.Module):
    """
    CNN for reconstructing 3D data.
    """
    
    def __init__(self, num_input_channels, num_output_channels):
        # Call super constructer
        super(ReconstructionCNN3D, self).__init__()

        # Activation function
        self.relu = nn.ReLU()

        # Series of convolutional layers
        self.conv1 = nn.Conv3d(num_input_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv3d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv3 = nn.Conv3d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv4 = nn.Conv3d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv5 = nn.Conv3d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv6 = nn.Conv3d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv7 = nn.Conv3d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv8 = nn.Conv3d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv9 = nn.Conv3d(48 * num_output_channels, 48 * num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)
        self.conv10 = nn.Conv3d(48 * num_output_channels, num_output_channels, kernel_size=7, stride=1, padding='same', bias=True)

        # Initialize weights in the network: all weights following Kaiming normal and all biases are 0
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu', a=0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Forward pass
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        return self.conv10(x)  
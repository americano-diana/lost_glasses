import torch
import torch.nn as nn

# --------------------------------------------------
# Flexible model architecture
# We can change output and downscaling factor
# --------------------------------------------------

class AlexNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        downscale: int | None = None,
    ):
        """
        AlexNet with optional width downscaling.

        Parameters
        ----------
        num_classes:
            Number of output classes.

        downscale:
            Width reduction factor.

            None or 1:
                Use the original channel and hidden-layer dimensions.

            2:
                Use half as many channels and hidden units.

            4:
                Use one quarter as many channels and hidden units.

        Examples
        --------
        Full-size model:
            model = AlexNet(num_classes=2)

        Half-width model:
            model = AlexNet(num_classes=2, downscale=2)

        Quarter-width model:
            model = AlexNet(num_classes=2, downscale=4)
        """
        super().__init__()

        # Treat None as no downscaling.
        if downscale is None:
            downscale = 1

        if not isinstance(downscale, int):
            raise TypeError(
                "downscale must be an integer or None."
            )

        if downscale < 1:
            raise ValueError(
                "downscale must be greater than or equal to 1."
            )

        self.num_classes = num_classes
        self.downscale = downscale

        # --------------------------------------------------
        # Calculate model dimensions
        # --------------------------------------------------

        retina_channels = max(1, 64 // downscale)
        lgn_channels = max(1, 192 // downscale)
        v1_channels = max(1, 384 // downscale)
        v2_channels = max(1, 256 // downscale)
        v4_channels = max(1, 256 // downscale)

        hidden_features = max(1, 4096 // downscale)

        # Save dimensions for inspection or logging.
        self.model_dimensions = {
            "retina_channels": retina_channels,
            "lgn_channels": lgn_channels,
            "v1_channels": v1_channels,
            "v2_channels": v2_channels,
            "v4_channels": v4_channels,
            "hidden_features": hidden_features,
        }

        # --------------------------------------------------
        # Convolutional layers
        # --------------------------------------------------

        self.retina = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=retina_channels,
                kernel_size=11,
                stride=4,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(retina_channels),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
        )

        self.lgn = nn.Sequential(
            nn.Conv2d(
                in_channels=retina_channels,
                out_channels=lgn_channels,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(lgn_channels),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
        )

        self.v1 = nn.Sequential(
            nn.Conv2d(
                in_channels=lgn_channels,
                out_channels=v1_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(v1_channels),
        )

        self.v2 = nn.Sequential(
            nn.Conv2d(
                in_channels=v1_channels,
                out_channels=v2_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(v2_channels),
        )

        self.v4 = nn.Sequential(
            nn.Conv2d(
                in_channels=v2_channels,
                out_channels=v4_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(v4_channels),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
        )

        # --------------------------------------------------
        # Fully connected layers
        # --------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        flattened_features = v4_channels * 6 * 6

        self.it = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                flattened_features,
                hidden_features,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(
                hidden_features,
                hidden_features,
            ),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(
            hidden_features,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the network.
        """
        x = self.retina(x)
        x = self.lgn(x)
        x = self.v1(x)
        x = self.v2(x)
        x = self.v4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.it(x)
        x = self.classifier(x)

        return x
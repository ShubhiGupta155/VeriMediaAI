"""
Network architectures for video deepfake detection.
"""

from modules.video.network.xception import Xception, SeparableConv2d, Block

__all__ = ["Xception", "SeparableConv2d", "Block"]

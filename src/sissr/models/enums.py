from __future__ import annotations

from enum import Enum


class ModelName(str, Enum):
    DIFFSSR = "diffssr"
    STEREO_SR = "stereo_sr"


class ResiConnection(str, Enum):
    ONE_CONV = "1conv"
    IDENTITY = "identity"


class ResidualStrategy(str, Enum):
    DEPTH_AGG = "depth_agg"
    STANDARD = "standard"


class CrossPosEncoding(str, Enum):
    NONE = "none"
    WINDOW_ROPE = "window_rope"
    EPIPOLAR_ROPE = "epipolar_rope"
    RECTIFIED_DISPARITY_ROPE = "rectified_disparity_rope"


class StereoDirection(int, Enum):
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = -1

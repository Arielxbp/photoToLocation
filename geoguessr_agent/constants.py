from __future__ import annotations

import numpy as np
import torch

IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_NET_MEAN_T = torch.tensor([0.485, 0.456, 0.406])
IMAGE_NET_STD_T = torch.tensor([0.229, 0.224, 0.225])

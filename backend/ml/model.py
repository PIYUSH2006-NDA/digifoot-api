# ---------------------------
# SAFE TORCH IMPORT (NEW)
# ---------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False


# ---------------------------
# SIMPLE CNN DEPTH MODEL
# ---------------------------
if TORCH_AVAILABLE:
    class FootDepthModel(nn.Module):
        def __init__(self):
            super(FootDepthModel, self).__init__()

            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),

                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU()
            )

            self.decoder = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),

                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(),

                nn.Conv2d(16, 1, 3, padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


# ---------------------------
# LOAD MODEL (SAFE)
# ---------------------------
def load_model(model_path=None):

    if not TORCH_AVAILABLE:
        print("⚠️ Torch not available — skipping ML model")
        return None

    model = FootDepthModel()

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()
    return model


# ---------------------------
# PREDICT DEPTH (SAFE)
# ---------------------------
def predict_depth(model, mask):

    if not TORCH_AVAILABLE or model is None:
        raise Exception("ML model not available")

    import numpy as np

    mask = mask.astype("float32")
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        depth = model(mask)

    depth = depth.squeeze().numpy()

    return depth
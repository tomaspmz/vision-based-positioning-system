"""
VBAPS Model — SeCo-pretrained ResNet-50 with frozen backbone.

SeCo (Seasonal Contrast) provides self-supervised weights trained on
Sentinel-2 imagery, giving a strong remote-sensing feature extractor.

Checkpoint source:
  https://zenodo.org/records/4728033/files/seco_resnet50_1m.ckpt

The checkpoint is in PyTorch-Lightning format.  This module handles
the key remapping so it loads cleanly into a standard torchvision
ResNet-50.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
SECO_FILENAME = "seco_resnet50_1m.ckpt"
SECO_PATH = os.path.join(CKPT_DIR, SECO_FILENAME)
SECO_URL = "https://zenodo.org/records/4728033/files/seco_resnet50_1m.ckpt"


def _download_seco():
    """Download the SeCo checkpoint if not already cached."""
    if os.path.isfile(SECO_PATH):
        return
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[model] Downloading SeCo checkpoint to {SECO_PATH} ...")
    torch.hub.download_url_to_file(SECO_URL, SECO_PATH)
    print("[model] Download complete.")


def _load_seco_state_dict():
    """Load and remap SeCo Lightning checkpoint → torchvision ResNet-50 keys.

    The .ckpt was saved by an old PyTorch Lightning version.  Rather than
    depending on a specific PL release, we use a custom Unpickler that
    stubs out any unknown PL classes so we can grab the state_dict tensor
    data without actually importing pytorch-lightning.
    """
    import pickle
    import zipfile
    import io

    _download_seco()

    # ── Custom unpickler that stubs missing PL modules ────────────────
    class _PLStubUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, ImportError, AttributeError):
                # Return a no-op class so unpickling succeeds
                return type(name, (), {"__reduce__": lambda self: (type(self), ())})

    # torch.load with a custom pickle_module
    class _PickleShim:
        Unpickler = _PLStubUnpickler
        load = staticmethod(pickle.load)
        dumps = staticmethod(pickle.dumps)

    ckpt = torch.load(SECO_PATH, map_location="cpu", weights_only=False,
                       pickle_module=_PickleShim)
    raw = ckpt["state_dict"]

    # SeCo uses MoCo-style training with encoder_q (query) & encoder_k (key).
    # We want encoder_q — the query encoder.
    # Keys are Sequential-indexed: encoder_q.{0,1,4,5,6,7} → ResNet named layers.
    IDX_TO_NAME = {
        "0": "conv1",    # Conv2d(3,64,7,2,3)
        "1": "bn1",      # BatchNorm2d(64)
        # 2 = relu, 3 = maxpool (no params)
        "4": "layer1",
        "5": "layer2",
        "6": "layer3",
        "7": "layer4",
    }

    cleaned = {}
    for k, v in raw.items():
        if not k.startswith("encoder_q."):
            continue
        rest = k[len("encoder_q."):]       # e.g. "4.0.bn1.weight"
        parts = rest.split(".", 1)
        seq_idx = parts[0]                  # e.g. "4"
        if seq_idx not in IDX_TO_NAME:
            continue
        layer_name = IDX_TO_NAME[seq_idx]   # e.g. "layer1"
        if len(parts) == 1:
            # conv1 weight lives at encoder_q.0.weight → conv1.weight
            new_key = layer_name + ".weight"
        else:
            new_key = layer_name + "." + parts[1]
        cleaned[new_key] = v

    return cleaned


def build_model(num_classes: int, freeze_backbone: bool = True):
    """
    Return a ResNet-50 with:
      - SeCo-pretrained backbone (optionally frozen)
      - Fresh classification head:  Linear(2048, num_classes)
    """
    model = models.resnet50(weights=None)

    # Load SeCo weights into the backbone
    seco_sd = _load_seco_state_dict()
    missing, unexpected = model.load_state_dict(seco_sd, strict=False)
    # 'fc.weight' and 'fc.bias' will be in missing — that's expected
    print(f"[model] SeCo weights loaded.  missing={len(missing)}  "
          f"unexpected={len(unexpected)}")

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze backbone (everything except fc)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[model] Backbone frozen.  Trainable: {trainable:,} / {total:,}")

    return model

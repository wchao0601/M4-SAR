# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""
from .dynamic import CMA # M4-SAR
from .fusion import OS_KANFusion, MambaFusionBlock, FeatureFusionModule, CMIM # M4-SAR
from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    SilenceRadar, # M4-SAR
    SilenceRGB,   # M4-SAR
    GPT,          # M4-SAR
    Add_GPT,      # M4-SAR
    CSSA,         # M4-SAR
    OSR,          # M4-SAR
    GPTcross,     # M4-SAR
    ICAFusion,    # M4-SAR
    MMI_fourier,  # M4-SAR
    AAttenADD, # M4-SAR
    MLLADD, # M4-SAR
    SilenceGrad, # M4-SAR
    SilenceHOG, # M4-SAR
    SilenceWST, # M4-SAR
    SilenceSAR, # M4-SAR
    SilenceRGBHOG, # M4-SAR
    FAM_OPT, # M4-SAR
    FAM_SAR, # M4-SAR
    AFM, # M4-SAR
    
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    Conadd,  # M4-SAR
    Conacbam # M4-SAR
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "v10Detect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SilenceRGB",   # M4-SAR
    "SilenceRadar", # M4-SAR
    "Conadd",       # M4-SAR
    "GPT",          # M4-SAR
    "Add_GPT",      # M4-SAR
    "CSSA",         # M4-SAR
    "Conacbam",     # M4-SAR
    "CMA",          # M4-SAR
    "OSR",          # M4-SAR
    "OS_KANFusion", # M4-SAR
    "GPTcross",     # M4-SAR
    "ICAFusion",    # M4-SAR
    "MMI_fourier",  # M4-SAR
    "AAttenADD", # M4-SAR
    "FeatureFusionModule", # M4-SAR
    "MambaFusionBlock", # M4-SAR
    "MLLADD", # M4-SAR
    "SilenceGrad", # M4-SAR
    "SilenceHOG", # M4-SAR
    "SilenceWST", # M4-SAR
    "SilenceSAR", # M4-SAR
    "SilenceRGBHOG", # M4-SAR
    "FAM_OPT", # M4-SAR
    "FAM_SAR", # M4-SAR
    "CMIM", # M4-SAR
    "AFM", # M4-SAR
    )
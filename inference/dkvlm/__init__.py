"""DK-VLM: a vision-language model that reasons over explicit 3D structure.

A frozen feed-forward reconstruction model turns the input frames into a world point map
and camera poses. Those are attached to the language model's visual tokens and consumed by
a geometry attention branch, so spatial relations are read off geometry instead of being
inferred from pixels alone.

Typical use::

    from dkvlm import load_dkvlm

    model = load_dkvlm(
        "../ckpt/DKVLM_Qwen3_8B",
        base_model_path="Qwen/Qwen3-VL-8B-Instruct",
        geometry_ckpt="weights/vggt_omega_1b_512.pt",
    )
    print(model.answer("scene.mp4", "How many chairs are in the room?"))
"""

from .frames import SampledFrames, load_image_frames, sample_video_frames
from .geometry import GeometryBundle, VGGTOmegaConfig, VGGTOmegaEngine
from .loader import GeoBranchConfig, DKVLM, build_prompt, load_dkvlm
from .qwen3_geo import DKVLMGeoWrapper, install_geo_attention, uninstall_geo_attention
from .tokens import build_geo_coords

__version__ = "1.0.0"

__all__ = [
    "GeoBranchConfig",
    "GeometryBundle",
    "DKVLM",
    "DKVLMGeoWrapper",
    "SampledFrames",
    "VGGTOmegaConfig",
    "VGGTOmegaEngine",
    "build_geo_coords",
    "build_prompt",
    "install_geo_attention",
    "load_image_frames",
    "load_dkvlm",
    "sample_video_frames",
    "uninstall_geo_attention",
]

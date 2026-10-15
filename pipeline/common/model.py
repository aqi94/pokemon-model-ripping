"""Game-agnostic intermediate representation (IR).

Every game importer produces a `Model`; everything downstream (GLB writer,
manifest writer, Blender export, viewer) consumes only this. Nothing here may
reference a specific game format.

Conventions:
  * Right-handed, Y up, metres.
  * Rotations are quaternions (x, y, z, w).
  * Joint `inverse_bind` is a 4x4 row-major numpy array (column vectors,
    i.e. glTF convention when flattened column-major on export).
  * Textures are RGBA8 numpy arrays of shape (H, W, 4), top row first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

TrackKind = Literal["translation", "rotation", "scale"]
AlphaMode = Literal["OPAQUE", "MASK", "BLEND"]


@dataclass
class Joint:
    name: str
    parent: int  # index into Skeleton.joints, -1 for root
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    inverse_bind: np.ndarray | None = None  # 4x4, only for skinning joints
    skinning: bool = False  # participates in the skin palette


@dataclass
class Skeleton:
    joints: list[Joint]
    # Order of joints as referenced by vertex joint indices. Entry i is an
    # index into `joints`. Importers map their native palette to this.
    palette: list[int]

    def index_of(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        return -1


@dataclass
class TextureImage:
    key: str  # unique within the model, used as file stem
    rgba: np.ndarray  # (H, W, 4) uint8
    srgb: bool = True  # colour data (True) vs linear data (False)

    @property
    def width(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgba.shape[0])


@dataclass
class Material:
    name: str
    base_color: str | None = None  # TextureImage.key
    normal: str | None = None
    occlusion: str | None = None
    emissive: str | None = None
    metallic_roughness: str | None = None
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metallic_factor: float = 0.0
    roughness_factor: float = 0.8
    alpha_mode: AlphaMode = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    uv_sets: dict[str, int] = field(default_factory=dict)  # slot -> TEXCOORD index
    # Texture wrap per slot: (wrapS, wrapT) in {"repeat", "mirror", "clamp"}
    wrap: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Extra UV transform per slot on top of the primitive UVs:
    # (scale_u, scale_v, offset_u, offset_v), emitted as KHR_texture_transform.
    uv_transform: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


@dataclass
class Primitive:
    material: str  # Material.name
    indices: np.ndarray  # (N,) uint32, triangles
    positions: np.ndarray  # (V, 3) float32
    normals: np.ndarray | None = None  # (V, 3)
    tangents: np.ndarray | None = None  # (V, 4)
    uvs: list[np.ndarray] = field(default_factory=list)  # each (V, 2)
    colors: np.ndarray | None = None  # (V, 4) float32 0..1
    joints: np.ndarray | None = None  # (V, 4) uint16, indices into Skeleton.palette
    weights: np.ndarray | None = None  # (V, 4) float32


@dataclass
class Mesh:
    name: str
    primitives: list[Primitive]
    lod: int = 0  # 0 = full detail
    visible: bool = True  # rest-pose visibility (some shapes are alt states)
    extras: dict = field(default_factory=dict)


@dataclass
class Track:
    joint: str
    kind: TrackKind
    times: np.ndarray  # (K,) float32 seconds
    values: np.ndarray  # (K, 3) or (K, 4) float32


@dataclass
class Clip:
    name: str
    tracks: list[Track]
    frame_rate: float = 30.0
    frame_count: int = 0
    loop: bool = False
    index: int = 0  # sort key / native id
    category: str = "other"
    group: str = ""  # alternate animation set the clip belongs to ("" = main)
    source: str = ""
    # Per-mesh visibility keys: mesh name -> list of (frame, visible). Meshes
    # absent from the dict keep their rest visibility during this clip.
    visibility: dict[str, list[tuple[int, bool]]] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.frame_count > 1 and self.frame_rate > 0:
            return (self.frame_count - 1) / self.frame_rate
        return max((float(t.times[-1]) for t in self.tracks if len(t.times)), default=0.0)


@dataclass
class Variant:
    """An alternate colouring (shiny, chroma, outfit colour).

    Variants are organised in independent `group`s so a model can expose
    several colour axes at once (e.g. group "coloring" = normal/shiny and
    group "skin" = tone A/B). The viewer shows one swatch row per group and
    the selected option of every group is applied together. Within a group,
    the option with `default=True` (or the first) is the rest state.
    """

    name: str
    materials: dict[str, Material]  # overrides, keyed by Material.name
    group: str = "coloring"
    label: str = ""
    default: bool = False


@dataclass
class Model:
    id: str
    game: str
    display_name: str
    kind: str  # "pokemon" | "trainer" | "object"
    skeleton: Skeleton | None
    meshes: list[Mesh]
    materials: dict[str, Material]
    textures: dict[str, TextureImage]
    clips: list[Clip] = field(default_factory=list)
    variants: list[Variant] = field(default_factory=list)
    default_clip: str | None = None
    source_paths: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def material_for(self, name: str, variant: str | None = None) -> Material:
        if variant:
            for v in self.variants:
                if v.name == variant and name in v.materials:
                    return v.materials[name]
        return self.materials[name]

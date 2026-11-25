"""stitcher: the stitcher of the Helsinki lab."""

from typing import Annotated

import numpy as np
import xarray as xr

from arkitekt import App, Description, Task
from mikro import Mikro, dataset_arrays
from mikro.api.schema import AxisType
from mikro.arkitekt.specs import RGBImage

# =========================================================================== #
# stitcher: the stitcher                                                      #
# =========================================================================== #
# Tiles side by side in the order given. No registration, no blending.
stitcher_app = App(
    "stitcher",
    "0.1.0",
    description="Stitches image tiles into one mosaic.",
)


@stitcher_app.action
def stitch_images(
    mikro: Mikro,
    images: Annotated[list[RGBImage], Description("The tiles, left to right")],
    *,
    task: Task,
) -> RGBImage:
    """Stitch Images

    Joins a row of equally sized RGB tiles into one image
    """
    if not images:
        raise ValueError("Nothing to stitch: no tiles were given")

    tiles: list[np.ndarray] = []
    derived_from = []
    offset = 0
    for index, image in enumerate(images):
        task.progress(
            int(70 * (index + 1) / len(images)),
            f"Reading tile {index + 1}/{len(images)}",
        )
        # to (c, y, x)
        channel = image.axes_of_type(AxisType.CHANNEL)[0]
        pixels = xr.DataArray(np.asarray(image.data), dims=list(image.axis_names))
        pixels = pixels.rename({channel: "c"}).transpose("c", ...).values
        if pixels.ndim != 3:
            raise ValueError(f"Tile {index} is not an RGB plane: shape {pixels.shape}")
        if tiles and pixels.shape[1] != tiles[0].shape[1]:
            raise ValueError(
                f"Tile {index} is {pixels.shape[1]} px high, the first is {tiles[0].shape[1]}"
            )
        tiles.append(pixels)
        # mosaic -> tile: the mosaic origin is `offset` px left of this tile
        derived_from.append(
            image.derive_translation(
                [0.0, 0.0, -float(offset)], value_relation="IDENTICAL"
            )
        )
        offset += pixels.shape[2]

    task.progress(80, "Storing the mosaic")
    mosaic = xr.DataArray(np.concatenate(tiles, axis=2), dims=("c", "y", "x"))
    # same pyramid as the tiles
    data, scales = dataset_arrays(mosaic, levels=4, method="mean")
    dataset = mikro.create_array_dataset(
        data=data,
        scales=scales,
        name=f"stitched ({len(tiles)} tiles)",
        axes=["c", "y", "x"],
        derived_from=derived_from,
    )
    # No recipe: the values are IDENTICAL to the tiles', so the server draws it like them
    # (one RGB layer).
    dataset.stage(name=f"stitched ({len(tiles)} tiles)")
    print(f"[stitcher] stitched {len(tiles)} tiles into {dataset.id} {mosaic.shape}")
    return dataset.lens()


#: Logo: four tiles, stitched seams.
STITCHER_LOGO = """
<Blok kind="agent" context="logo">
  <Logo viewBox="0 0 64 64" title="Stitcher">
    <Defs>
      <LinearGradient id="stitch-ground" x1="0" y1="0" x2="1" y2="1">
        <Stop offset="0" stopColor="ink" />
        <Stop offset="1" stopColor="brand-strong" />
      </LinearGradient>
      <RadialGradient id="stitch-glow" cx="0.5" cy="0.45" r="0.6">
        <Stop offset="0" stopColor="spectrum-violet" stopOpacity="0.45" />
        <Stop offset="1" stopColor="spectrum-violet" stopOpacity="0" />
      </RadialGradient>
      <ClipPath id="stitch-tile"><Rect x="0" y="0" width="64" height="64" rx="14" /></ClipPath>
    </Defs>
    <G clipPath="url(#stitch-tile)">
      <Rect x="0" y="0" width="64" height="64" fill="url(#stitch-ground)" />
      <Rect x="0" y="0" width="64" height="64" fill="url(#stitch-glow)" />
      <G fill="surface">
        <Rect x="11" y="11" width="20" height="20" rx="2.5" fillOpacity="0.9" />
        <Rect x="33" y="11" width="20" height="20" rx="2.5" fillOpacity="0.75" />
        <Rect x="11" y="33" width="20" height="20" rx="2.5" fillOpacity="0.75" />
        <Rect x="33" y="33" width="20" height="20" rx="2.5" fillOpacity="0.9" />
      </G>
      <G fill="spectrum-violet">
        <Ellipse cx="32" cy="21" rx="5" ry="3.5" />
        <Ellipse cx="21" cy="40" rx="3" ry="2.5" fillOpacity="0.8" />
        <Ellipse cx="42" cy="44" rx="4" ry="5" fillOpacity="0.9" />
        <Ellipse cx="32" cy="32" rx="2.5" ry="2.5" fillOpacity="0.7" />
        <Ellipse cx="18" cy="18" rx="2" ry="1.6" fillOpacity="0.6" />
      </G>
      <G stroke="spectrum-violet" strokeWidth="1.6" strokeDasharray="2 2" strokeLinecap="round" animate="march">
        <Line x1="32" y1="8" x2="32" y2="56" />
        <Line x1="8" y1="32" x2="56" y2="32" />
      </G>
    </G>
    <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
  </Logo>
</Blok>
"""

stitcher_app.blok(
    "stitcher-logo",
    STITCHER_LOGO,
    description="Logo.",
    default=True,
)

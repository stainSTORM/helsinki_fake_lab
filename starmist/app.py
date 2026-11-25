"""starmist: the segmenter of the Helsinki lab."""

from typing import Annotated

import numpy as np
import xarray as xr
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.measure import label

from arkitekt import App, Description, Task
from mikro import Mikro
from mikro.api.schema import AxisType
from mikro.arkitekt.specs import LabelMask, RGBImage

# =========================================================================== #
# starmist: the segmenter                                                     #
# =========================================================================== #
# No model: colour deconvolution, Otsu and connected components. Hematoxylin is high where
# the nuclei are.
starmist_app = App(
    "starmist",
    "0.1.0",
    description="Segments cells in an image.",
)


@starmist_app.action
def segment_image(
    mikro: Mikro,
    image: Annotated[RGBImage, Description("The H&E image to segment")],
    *,
    task: Task,
) -> LabelMask:
    """Segment Image

    Labels the nuclei of an H&E image and stores the labels as a new dataset
    """
    task.progress(10, "Reading the image")
    channel = image.axes_of_type(AxisType.CHANNEL)[0]
    plane = [name for name in image.axis_names if name != channel]
    pixels = xr.DataArray(np.asarray(image.data), dims=list(image.axis_names))
    rgb = pixels.transpose(*plane, channel).values

    task.progress(50, "Labelling")
    hematoxylin = rgb2hed(rgb)[..., 0]
    labels = xr.DataArray(
        label(hematoxylin > threshold_otsu(hematoxylin)).astype(np.uint16),
        dims=plane,
    )

    task.progress(80, "Storing the labels")
    result = mikro.create_array_dataset(
        data=labels,
        scales=[],
        name="nuclei",
        axes=image.carried_axes(plane),
        derived_from=[image.derive_projection(plane, value_relation="CATEGORIZED")],
    )
    print(f"[starmist] {int(labels.max())} nuclei in {result.id}")
    return result.lens()


#: Logo: star-convex cell outlines.
STARMIST_LOGO = """
<Blok kind="agent" context="logo">
  <Logo viewBox="0 0 64 64" title="Starmist">
    <Defs>
      <LinearGradient id="star-ground" x1="0" y1="0" x2="1" y2="1">
        <Stop offset="0" stopColor="ink" />
        <Stop offset="1" stopColor="brand-strong" />
      </LinearGradient>
      <RadialGradient id="star-glow" cx="0.5" cy="0.45" r="0.6">
        <Stop offset="0" stopColor="spectrum-green" stopOpacity="0.45" />
        <Stop offset="1" stopColor="spectrum-green" stopOpacity="0" />
      </RadialGradient>
      <ClipPath id="star-tile"><Rect x="0" y="0" width="64" height="64" rx="14" /></ClipPath>
    </Defs>
    <G clipPath="url(#star-tile)">
      <Rect x="0" y="0" width="64" height="64" fill="url(#star-ground)" />
      <Rect x="0" y="0" width="64" height="64" fill="url(#star-glow)" />
      <G fill="spectrum-green" fillOpacity="0.22" stroke="spectrum-green" strokeWidth="1.6" strokeLinejoin="round">
        <Polygon points="38.7,29.6 32.1,36.2 23.6,38.8 14.3,34.7 14.7,24.7 20.5,18.7 28.5,14.7 35.2,20.9" />
        <Polygon points="52.9,45.3 47.3,48.7 40.4,49.3 36.8,43.4 38.4,37.4 42.5,32.4 48.8,34.0 53.2,38.6" />
        <Polygon points="25.5,48.6 22.8,52.6 18.3,55.0 14.4,51.8 13.5,47.5 14.9,43.0 19.6,42.0 24.4,43.6" />
      </G>
      <Path d="M26 27L38.7 29.6M26 27L32.1 36.2M26 27L23.6 38.8M26 27L14.3 34.7M26 27L14.7 24.7M26 27L20.5 18.7M26 27L28.5 14.7M26 27L35.2 20.9" stroke="surface" strokeOpacity="0.55" strokeWidth="0.9" strokeLinecap="round" animate="draw" />
      <Circle cx="26" cy="27" r="2.2" fill="surface" />
      <Circle cx="45.5" cy="41" r="1.6" fill="surface" fillOpacity="0.85" />
      <Circle cx="19.2" cy="48.5" r="1.3" fill="surface" fillOpacity="0.85" />
    </G>
    <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
  </Logo>
</Blok>
"""

starmist_app.blok(
    "starmist-logo",
    STARMIST_LOGO,
    description="Logo.",
    default=True,
)

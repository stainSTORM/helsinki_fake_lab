"""analyzer: the stain analyzer of the Helsinki lab."""

from typing import Annotated

import numpy as np
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table

from arkitekt import App, Description, Task
from mikro import Mikro
from mikro.api.schema import (
    ArrayLens,
    AxisType,
    ColumnInput,
    ColumnRole,
    DatasetIdentifiesInput,
    TableLens,
)
from mikro.arkitekt.specs import LabelMask, RGBImage

# =========================================================================== #
# analyzer: the stain analyzer                                                #
# =========================================================================== #
# `measure_cells` gives one row per cell. `calculate_stain_percentage` is the older one: how
# much of the image the labels cover.
analyzer_app = App(
    "analyzer",
    "0.1.0",
    description="Measures how much of a slide is stained.",
)

#: Long names of the cell table's columns.
CELL_COLUMNS = {
    "area_px": "area (pixels)",
    "area_um2": "area (µm²)",
    "centroid_y": "centroid y (pixels)",
    "centroid_x": "centroid x (pixels)",
    "bbox_min_y": "bounding box, top (pixels)",
    "bbox_min_x": "bounding box, left (pixels)",
    "bbox_max_y": "bounding box, bottom (pixels)",
    "bbox_max_x": "bounding box, right (pixels)",
    "eccentricity": "eccentricity",
    "solidity": "solidity",
    "mean_red": "mean red (0 to 255)",
    "mean_green": "mean green (0 to 255)",
    "mean_blue": "mean blue (0 to 255)",
    "mean_hematoxylin": "mean hematoxylin (optical density)",
    "mean_eosin": "mean eosin (optical density)",
    "stain_percentage": "stained share of the cell (%)",
}


def _plane(lens: ArrayLens, plane: list[str], *rest: str) -> np.ndarray:
    """The lens' pixels with its axes in the order `plane`, then `rest`."""
    data = lens.xarray_data
    dropped = [d for d in data.dims if d not in (*plane, *rest)]
    return np.asarray(data.squeeze(dropped).transpose(*plane, *rest).data)


def _pixel_area_um2(lens: ArrayLens, plane: list[str]) -> float | None:
    """How many µm² one pixel of `lens` covers, or None when it is not calibrated in µm."""
    sizes = {
        entry.axis: entry.size
        for entry in lens.voxel_size or ()
        if str(getattr(entry.type, "value", entry.type)) == "SPACE"
        and entry.unit in ("micrometer", "µm", "um")
    }
    if not all(axis in sizes for axis in plane):
        return None
    return float(np.prod([sizes[axis] for axis in plane]))


def cell_table(
    rgb: np.ndarray, labels: np.ndarray, pixel_area_um2: float | None
) -> tuple[dict[str, np.ndarray], float]:
    """Cell table columns for (y, x) `labels` over (y, x, c) uint8 `rgb`, plus the stained
    share of all cell pixels.

    Stained means hematoxylin above the image's Otsu threshold, the same rule starmist uses.
    """
    hed = rgb2hed(rgb)
    hematoxylin, eosin = hed[..., 0], hed[..., 1]
    stained = hematoxylin > threshold_otsu(hematoxylin)
    channels = np.stack(
        [
            *np.moveaxis(rgb.astype(np.float64), -1, 0),
            hematoxylin,
            eosin,
            stained.astype(np.float64),
        ],
        axis=-1,
    )
    props = regionprops_table(
        labels,
        intensity_image=channels,
        properties=(
            "label",
            "area",
            "centroid",
            "bbox",
            "eccentricity",
            "solidity",
            "intensity_mean",
        ),
    )
    area = props["area"].astype(np.int64)
    columns: dict[str, np.ndarray] = {
        "instance_id": props["label"].astype(np.int64),
        "area_px": area,
    }
    if pixel_area_um2 is not None:
        columns["area_um2"] = area * pixel_area_um2
    columns |= {
        "centroid_y": props["centroid-0"].astype(np.float64),
        "centroid_x": props["centroid-1"].astype(np.float64),
        "bbox_min_y": props["bbox-0"].astype(np.int64),
        "bbox_min_x": props["bbox-1"].astype(np.int64),
        "bbox_max_y": props["bbox-2"].astype(np.int64),
        "bbox_max_x": props["bbox-3"].astype(np.int64),
        "eccentricity": props["eccentricity"].astype(np.float64),
        "solidity": props["solidity"].astype(np.float64),
        "mean_red": props["intensity_mean-0"].astype(np.float64),
        "mean_green": props["intensity_mean-1"].astype(np.float64),
        "mean_blue": props["intensity_mean-2"].astype(np.float64),
        "mean_hematoxylin": props["intensity_mean-3"].astype(np.float64),
        "mean_eosin": props["intensity_mean-4"].astype(np.float64),
        "stain_percentage": 100 * props["intensity_mean-5"].astype(np.float64),
    }
    in_cells = labels > 0
    stained_share = float(stained[in_cells].mean()) if in_cells.any() else 0.0
    return columns, stained_share


@analyzer_app.action
def measure_cells(
    mikro: Mikro,
    image: Annotated[RGBImage, Description("The H&E image the cells were found in")],
    labels: Annotated[
        LabelMask, Description("The cells, each under its own id, on the image's grid")
    ],
    *,
    task: Task,
) -> tuple[TableLens, float]:
    """Measure Cells

    Measures every cell of the labels in the image: its size and shape, its mean
    colour, and how much of it is stained (its pixels whose hematoxylin is above the
    image's Otsu threshold)

    Returns
    -------
    TableLens
        One row per cell, keyed by its instance id in the labels
    float
        The stained share of all cell pixels, 0 to 1
    """
    task.progress(10, "Reading the image and the labels")
    (channel,) = image.axes_of_type(AxisType.CHANNEL)
    plane = [str(axis) for axis in image.axes_of_type(AxisType.SPACE)]
    rgb = _plane(image, plane, channel)
    cells = _plane(labels, plane)
    if cells.shape != rgb.shape[:2]:
        raise ValueError(
            f"The labels are {cells.shape} pixels, the image {rgb.shape[:2]}: "
            "they have to be on the same grid"
        )

    task.progress(40, "Measuring the cells")
    columns, stained_share = cell_table(
        rgb.astype(np.uint8), cells.astype(np.int64), _pixel_area_um2(image, plane)
    )

    task.progress(80, "Storing the table")
    table = mikro.create_table_dataset(
        name=f"Cells of {labels.name}" if labels.name else "Cells",
        data=columns,
        columns=[
            # rows are keyed by the cell's id in the labels
            ColumnInput(
                name="instance_id",
                axis_type=AxisType.INDEX,
                long_name="cell id",
                identified_by=(DatasetIdentifiesInput(dataset=labels.dataset.id),),
            ),
            *(
                ColumnInput(
                    name=name, role=ColumnRole.ATTRIBUTE, long_name=CELL_COLUMNS[name]
                )
                for name in columns
                if name != "instance_id"
            ),
        ],
    )
    print(
        f"[analyzer] {len(columns['instance_id'])} cells, {stained_share:.1%} of their pixels stained"
    )
    return table.lens(), stained_share


@analyzer_app.action
def calculate_stain_percentage(
    mikro: Mikro,
    image: Annotated[LabelMask, Description("The labels of the stained objects")],
    *,
    task: Task,
) -> float:
    """Calculate Stain Percentage

    The fraction (0 to 1) of the image covered by labelled objects
    """
    task.progress(30, "Reading the labels")
    # `mikro` is injected only so the labels' zarr store is read with its client.
    labels = np.asarray(image.data)
    fraction = float(np.count_nonzero(labels)) / max(1, labels.size)
    print(f"[analyzer] {fraction:.1%} stained")
    return fraction


#: Logo: a stain gauge over bars.
ANALYZER_LOGO = """
<Blok kind="agent" context="logo">
  <Logo viewBox="0 0 64 64" title="Analyzer">
    <Defs>
      <LinearGradient id="gauge-ground" x1="0" y1="0" x2="1" y2="1">
        <Stop offset="0" stopColor="ink" />
        <Stop offset="1" stopColor="brand-strong" />
      </LinearGradient>
      <RadialGradient id="gauge-glow" cx="0.5" cy="0.45" r="0.6">
        <Stop offset="0" stopColor="spectrum-yellow" stopOpacity="0.45" />
        <Stop offset="1" stopColor="spectrum-yellow" stopOpacity="0" />
      </RadialGradient>
      <ClipPath id="gauge-tile"><Rect x="0" y="0" width="64" height="64" rx="14" /></ClipPath>
    </Defs>
    <G clipPath="url(#gauge-tile)">
      <Rect x="0" y="0" width="64" height="64" fill="url(#gauge-ground)" />
      <Rect x="0" y="0" width="64" height="64" fill="url(#gauge-glow)" />
      <Circle cx="32" cy="28" r="15" fill="none" stroke="surface" strokeOpacity="0.2" strokeWidth="6" />
      <Circle cx="32" cy="28" r="15" fill="none" stroke="spectrum-yellow" strokeWidth="6" strokeLinecap="round" strokeDasharray="30 95" transform="rotate(-90 32 28)" animate="draw" />
      <SvgText x="32" y="28.5" text="%" fill="surface" fontSize="12" fontWeight="700" textAnchor="middle" dominantBaseline="central" />
      <G fill="surface">
        <Rect x="17" y="50" width="6" height="5" rx="1" fillOpacity="0.55" />
        <Rect x="25" y="46" width="6" height="9" rx="1" fillOpacity="0.75" />
        <Rect x="41" y="48" width="6" height="7" rx="1" fillOpacity="0.55" />
      </G>
      <Rect x="33" y="43" width="6" height="12" rx="1" fill="spectrum-yellow" />
    </G>
    <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
  </Logo>
</Blok>
"""

analyzer_app.blok(
    "analyzer-logo",
    ANALYZER_LOGO,
    description="Logo.",
    default=True,
)

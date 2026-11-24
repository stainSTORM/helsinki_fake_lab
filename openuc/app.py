"""openuc: the microscope of the Helsinki lab.

Shows a real slide in brightfield: the Visium HD mouse small intestine H&E ($OPENUC_SLIDE).
The live view streams through lovekit; stage, focus and exposure change it. acquire_image
and snap store the current view in mikro as an RGB image with its own scene.
"""

import asyncio
import functools
import os
import time
from dataclasses import field
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import xarray as xr
import zarr

import koil
from arkitekt import App, Description, Task
from kanne.scalars import Unit
from lovekit import VideoStream, lovekit_service
from lovekit.api.schema import Stream
from lovekit.lovekit import Lovekit
from mikro import Mikro, dataset_arrays, mikro_service, space_2d
from mikro.api.schema import ArrayDataset, ArrayLens, CoordinateAnchorInput, Scene
from mikro.arkitekt.specs import RGBImage

# =========================================================================== #
# openuc: the microscope                                                      #
# =========================================================================== #
openuc_app = App(
    "openuc",
    "0.1.0",
    description="openUC2 FRAME microscope: stage, focus, live view and frames into mikro.",
    services=[mikro_service, lovekit_service],
)

# openUC2 FRAME in metres, Draco-compressed. Reduced from testdata/frame3.glb by
# testdata/frame3_lite.mjs (2.2 M -> 207 k triangles; frame3_lite_fine.glb is a 384 k
# variant). ``stage_x`` > ``stage_y`` and ``light`` stay separate nodes for the scene to
# bind.
openuc_app.register_asset("frame", "models/frame3_lite.glb")

#: stage travel either way, in metres (the slide's edge)
STAGE_LIMIT = 0.0033

#: focus travel either way, in µm
FOCUS_LIMIT = 200.0

#: Tile row starts in µm from the stage centre (y, x), picked on the thumbnail where tissue
#: fills three views. The slide name picks the row.
TILE_ROWS = [(-1995.0, -390.0), (2245.0, -1717.0), (-313.0, -2247.0)]


def _stage_coordinates(position: str) -> tuple[float, float]:
    """Stage position for a name, in metres from the centre.

    "home" is the centre. ``slide A_tile_4`` is tile 4 of slide A's row, one view to the
    right per tile. Anything else is the start of its row.
    """
    if position == "home":
        return 0.0, 0.0
    stem, _, digits = position.rpartition("_")
    tile = int(digits) if digits.isdigit() and stem else 0
    stem = stem.removesuffix("_tile") if digits.isdigit() and stem else position
    # byte sum, not a hash: slide A/B/C get neighbouring rows
    y_um, x_um = TILE_ROWS[sum(stem.encode()) % len(TILE_ROWS)]
    return (x_um + tile * VIEW_WIDTH * UM_PER_PIXEL) * 1e-6, y_um * 1e-6


@openuc_app.state(name="stage")
class StageState:
    """Where the stage is, how many frames were taken, and whether one is being taken."""

    position: str = "home"
    #: metres from the centre
    x: float = 0.0
    y: float = 0.0
    #: µm, sharp at BEST_FOCUS
    z: float = 0.0
    frames_acquired: int = 0
    acquiring: bool = False
    #: on while a frame is exposed
    light_on: bool = False


@openuc_app.state(name="camera")
class CameraState:
    """Live view: on/off, exposure, and the lovekit stream (empty until declared)."""

    live: bool = False
    exposure_ms: float = 50.0
    streams: list[Stream] = field(default_factory=list)
    #: live is on and lovekit took the stream
    streaming: bool = False
    #: scene of the last frame taken
    last_scene: Scene | None = None


@openuc_app.startup
def initialize_stage() -> StageState:
    """Start at the home position."""
    print("[openuc] ready")
    return StageState()


@openuc_app.startup
def initialize_camera() -> CameraState:
    """Live view off; the stream is declared once the app runs."""
    return CameraState()


def _store_view(
    mikro: Mikro, stage: StageState, camera: CameraState, name: str, task: Task
) -> tuple[ArrayDataset, ArrayLens, Scene]:
    """Take a frame at the stage position into mikro. Returns dataset, lens and scene."""
    took: dict[str, float] = {}
    mark = time.perf_counter()

    def step(label: str) -> None:
        nonlocal mark
        now = time.perf_counter()
        took[label] = now - mark
        mark = now

    stage.light_on = True
    task.progress(5, "Focusing")
    autofocus(stage)
    task.progress(25, "Exposing")
    stage.acquiring = True
    pixels = expose(stage.x, stage.y, stage.z, camera.exposure_ms, np.random.default_rng())
    koil.sleep(camera.exposure_ms / 1000)
    stage.acquiring = False
    stage.light_on = camera.live
    x_um, y_um = stage.x * 1e6, stage.y * 1e6
    step("expose")

    task.progress(40, "Storing the image")
    image = xr.DataArray(pixels.transpose(2, 0, 1), dims=("c", "y", "x"))
    # pyramid: 2048, 1024, 512, 256 px
    data, scales = dataset_arrays(image, levels=4, method="mean")
    dataset = mikro.create_array_dataset(
        data=data,
        scales=scales,
        name=name,
        axes=["c", "y", "x"],
        anchors=CoordinateAnchorInput.histogram_anchors(image, axis="c"),
    )
    stage.frames_acquired += 1
    step("upload")

    task.progress(70, "Placing it on the stage")
    # One stage space for every frame. The image's corner is half a view up-left of the
    # stage position.
    space = space_2d(mikro, "openUC2 stage", unit=Unit("micrometer"), reference="stage")
    space.register(
        dataset,
        scale={"y": UM_PER_PIXEL, "x": UM_PER_PIXEL},
        y=y_um - VIEW_HEIGHT / 2 * UM_PER_PIXEL,
        x=x_um - VIEW_WIDTH / 2 * UM_PER_PIXEL,
        mikro=mikro,
    )
    step("place")

    task.progress(90, "Making the scene")
    # one RGB layer over the full 8-bit range (50 ms = as scanned)
    scene = mikro.create_scene(name=name, coordinate_system=space.id)
    whole = dataset.lens(mikro=mikro)
    mikro.create_rgb_layer(
        lens=whole.id,
        scene=scene.id,
        red_lens=dataset.lens(mikro=mikro, c=0).id,
        green_lens=dataset.lens(mikro=mikro, c=1).id,
        blue_lens=dataset.lens(mikro=mikro, c=2).id,
        clim_min=0.0,
        clim_max=255.0,
    )
    camera.last_scene = scene
    step("scene")
    print("[openuc] " + name + ": " + ", ".join(f"{k} {v:.2f} s" for k, v in took.items()))
    return dataset, whole, scene


@openuc_app.action
def move_to_position(
    state: StageState,
    position: Annotated[str, Description("The named stage position to go to")],
) -> None:
    """Move To Position

    Moves the stage to a named position
    """
    # set first: the 3D view eases there while this waits
    state.x, state.y = _stage_coordinates(position)
    koil.sleep(0.3)
    state.position = position
    print(f"[openuc] stage at {position}")


@openuc_app.action
def acquire_image(
    mikro: Mikro, stage: StageState, camera: CameraState, *, task: Task
) -> RGBImage:
    """Acquire Image

    Takes one frame at the current stage position, stores it in mikro and makes a scene
    of it
    """
    dataset, whole, scene = _store_view(
        mikro, stage, camera, f"openuc {stage.position} #{stage.frames_acquired}", task
    )
    print(f"[openuc] acquired {dataset.id} at {stage.position}, scene {scene.id}")
    return whole


@openuc_app.action
def jog(
    state: StageState,
    dx_um: Annotated[
        float, Description("How far to move along x, in micrometres")
    ] = 0.0,
    dy_um: Annotated[
        float, Description("How far to move along y, in micrometres")
    ] = 0.0,
) -> None:
    """Jog Stage

    Moves the stage by a distance from where it is, stopping at the end of its travel
    """
    x = min(max(state.x + dx_um * 1e-6, -STAGE_LIMIT), STAGE_LIMIT)
    y = min(max(state.y + dy_um * 1e-6, -STAGE_LIMIT), STAGE_LIMIT)
    travel = max(abs(x - state.x), abs(y - state.y))
    state.x, state.y = x, y
    # 20 mm/s, like the live view
    koil.sleep(0.1 + travel / 0.02)
    state.position = f"{x * 1e3:+.3f}, {y * 1e3:+.3f} mm"


@openuc_app.action
def focus(
    state: StageState,
    dz_um: Annotated[
        float, Description("How far to move the focus, in micrometres; up is positive")
    ],
) -> None:
    """Focus

    Moves the focus drive up or down by a distance
    """
    state.z = min(max(state.z + dz_um, -FOCUS_LIMIT), FOCUS_LIMIT)
    koil.sleep(0.05 + abs(dz_um) / 400)


@openuc_app.action
def run_autofocus(stage: StageState) -> None:
    """Run Autofocus

    Brings the slide into focus when it is slightly out of it; further out, focus closer
    by hand first
    """
    off = abs(stage.z - BEST_FOCUS)
    if off > AUTOFOCUS_RANGE:
        raise ValueError(
            f"Too far out of focus for the autofocus ({off:.0f} µm, it reaches "
            f"{AUTOFOCUS_RANGE:g} µm): focus closer by hand first"
        )
    autofocus(stage)
    # wait for the drive
    koil.sleep(0.3 + off / FOCUS_SPEED)


@openuc_app.action
def set_live(
    stage: StageState,
    camera: CameraState,
    on: Annotated[bool, Description("Whether the live view runs")],
) -> None:
    """Set Live View

    Starts or stops the live view, and the illumination with it
    """
    camera.live = on
    stage.light_on = on


@openuc_app.action
def set_exposure(
    camera: CameraState,
    exposure_ms: Annotated[
        float, Description("How long each live frame is exposed, in milliseconds")
    ] = 50.0,
) -> None:
    """Set Exposure

    Sets how long each frame of the live view is exposed
    """
    camera.exposure_ms = min(max(exposure_ms, 1.0), 500.0)


@openuc_app.action
def snap(mikro: Mikro, stage: StageState, camera: CameraState, *, task: Task) -> Scene:
    """Snap

    Takes the field in view into mikro, placed in the stage space where it was taken,
    and makes a scene of it
    """
    x_um, y_um = stage.x * 1e6, stage.y * 1e6
    dataset, _, scene = _store_view(
        mikro, stage, camera, f"openuc snap {x_um:+.0f}, {y_um:+.0f} µm", task
    )
    print(f"[openuc] snapped {dataset.id} at {x_um:+.0f}, {y_um:+.0f} µm, scene {scene.id}")
    return scene


# --------------------------------------------------------------------------- #
# The slide and the live view                                                  #
# --------------------------------------------------------------------------- #
# The slide is read at pyramid level s2, centred on the stage centre. A view comes straight
# from the zarr chunks (~3 ms), blurred by defocus and scaled by exposure. Off the slide is
# bare glass. Frames go through lovekit, never through state: every state change is stored.
#: The H&E pyramid (a spatialdata image), outside the repo: ``$OPENUC_SLIDE`` or testing's copy.
SLIDE_STORE = Path(
    os.environ.get(
        "OPENUC_SLIDE",
        Path.home()
        / "Code/testing/testdata/visium_hd/data.zarr/images/Visium_HD_Mouse_Small_Intestine_full_image",
    )
)
SLIDE_LEVEL = "s1"
#: s0 is 0.2738 µm/px (see upload_visium_hd.py), s1 half that: a 10x objective on the 2048
#: px camera, 1.1 mm field. 500 µm = 913 px, the panel's scale bar.
UM_PER_PIXEL = 0.2738 * 2
#: bare glass at 50 ms
GLASS = (235, 235, 238)

LIVE_STREAM = "live"
LIVE_FPS = 10
#: sensor size in px
VIEW_WIDTH, VIEW_HEIGHT = 2048, 2048
#: Focus position where the slide is sharp, in µm. Not 0, so a fresh start is slightly out
#: of focus.
BEST_FOCUS = 18.0
#: blur sigma per µm of defocus
DEFOCUS_BLUR = 0.55
#: how fast the view follows: m/s and µm/s
STAGE_SPEED = 0.02
FOCUS_SPEED = 400.0
#: the autofocus only works within this many µm of focus
AUTOFOCUS_RANGE = 30.0


@functools.cache
def _slide() -> zarr.Array:
    """The slide's pixels, (c, y, x) uint8, opened once and read by the view."""
    if not SLIDE_STORE.exists():
        raise FileNotFoundError(f"The H&E slide is not at {SLIDE_STORE}")
    return zarr.open_array(SLIDE_STORE / SLIDE_LEVEL, mode="r")


@functools.cache
def _noise_bank() -> np.ndarray:
    """Read noise around 8 (sigma 2), bigger than a view. Frames sample it at a random
    offset.
    """
    rng = np.random.default_rng(7)
    shape = (VIEW_HEIGHT + 256, VIEW_WIDTH + 256, 3)
    return np.clip(rng.normal(8.0, 2.0, shape), 0, 16).astype(np.uint8)


def _slide_view(x: float, y: float, size: tuple[int, int] = (VIEW_HEIGHT, VIEW_WIDTH)) -> np.ndarray:
    """The part of the slide in view with the stage at (x, y) metres, (y, x, c) uint8,
    ``size`` pixels around the stage position."""
    slide = _slide()
    _, height, width = slide.shape
    rows, columns = size
    left = round(x * 1e6 / UM_PER_PIXEL + width / 2) - columns // 2
    top = round(y * 1e6 / UM_PER_PIXEL + height / 2) - rows // 2
    view = np.empty((rows, columns, 3), dtype=np.uint8)
    view[:] = GLASS
    y0, y1 = max(top, 0), min(top + rows, height)
    x0, x1 = max(left, 0), min(left + columns, width)
    if y0 < y1 and x0 < x1:
        view[y0 - top : y1 - top, x0 - left : x1 - left] = np.moveaxis(
            slide[:, y0:y1, x0:x1], 0, -1
        )
    return view


def _defocus(view: np.ndarray, z: float, *, exact: bool = False) -> np.ndarray:
    """``view`` blurred for focus z.

    Unless ``exact``, wide blurs are done at reduced size and upscaled: much cheaper, but
    the upscaling leaves kinks a contrast measure picks up.
    """
    sigma = abs(z - BEST_FOCUS) * DEFOCUS_BLUR / UM_PER_PIXEL
    if sigma < 0.3:
        return view
    if sigma <= 8 or exact:
        return cv2.GaussianBlur(view, (0, 0), sigma)
    factor = sigma / 4
    rows, columns = view.shape[:2]
    small = cv2.resize(
        view, (max(1, round(columns / factor)), max(1, round(rows / factor))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (columns, rows), interpolation=cv2.INTER_LINEAR)


def expose(
    x: float, y: float, z: float, exposure_ms: float, rng: np.random.Generator
) -> np.ndarray:
    """Camera frame at stage (x, y) metres and focus z: blurred, exposed, noisy. (y, x, c)
    uint8.
    """
    view = cv2.convertScaleAbs(_defocus(_slide_view(x, y), z), alpha=exposure_ms / 50.0)
    noise = _noise_bank()
    dy, dx = rng.integers(0, 256, 2)
    view = cv2.add(view, noise[dy : dy + VIEW_HEIGHT, dx : dx + VIEW_WIDTH])
    return cv2.subtract(view, 8)


def autofocus(stage: StageState) -> None:
    """Jump to BEST_FOCUS if within AUTOFOCUS_RANGE, else do nothing."""
    if abs(stage.z - BEST_FOCUS) <= AUTOFOCUS_RANGE:
        stage.z = BEST_FOCUS


def render_live_frame(
    x: float, y: float, z: float, exposure_ms: float, rng: np.random.Generator
) -> np.ndarray:
    """One live frame, as the (height, width, 4) uint8 a lovekit video track takes."""
    return cv2.cvtColor(expose(x, y, z, exposure_ms, rng), cv2.COLOR_RGB2RGBA)


def _towards(current: float, target: float, step: float) -> float:
    """``current`` moved towards ``target`` by at most ``step``."""
    return (
        target
        if abs(target - current) <= step
        else current + step * np.sign(target - current)
    )


@openuc_app.background
async def stream_live_view(
    lovekit: Lovekit, stage: StageState, camera: CameraState
) -> None:
    """Stream the live view for as long as the app runs, or until a newer run takes over.

    The view eases towards the stage and focus, so a jog pans instead of jumping. Frames
    are made in a thread: all apps of a run share this event loop.
    """
    rng = np.random.default_rng()
    declared = VideoStream(
        LIVE_STREAM,
        title="Live view",
        description="FRAME camera.",
        max_framerate=LIVE_FPS,
        # the default 3 Mbit/s smears the nuclei
        max_bitrate=12_000_000,
    )
    async with lovekit.publish([declared]) as published:
        camera.streams = list(published.streams)
        x, y, z = stage.x, stage.y, stage.z
        while not published.stopped.is_set():
            started = asyncio.get_running_loop().time()
            if camera.live:
                x = _towards(x, stage.x, STAGE_SPEED / LIVE_FPS)
                y = _towards(y, stage.y, STAGE_SPEED / LIVE_FPS)
                z = _towards(z, stage.z, FOCUS_SPEED / LIVE_FPS)
                frame = await asyncio.to_thread(
                    render_live_frame, x, y, z, camera.exposure_ms, rng
                )
                await published.video(LIVE_STREAM).send(frame)
            if camera.streaming != camera.live:
                camera.streaming = camera.live
            # a frame takes 30-70 ms, sleep the rest of the tick
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(1 / LIVE_FPS - elapsed, 0.01))
        if camera.streaming:
            camera.streaming = False


#: 3D scene. Model Y is up, so stage y maps to model Z. Travel is drawn 8x: 3 mm is nothing
#: on a 30 cm device.
FRAME_SCENE = """
<Blok kind="agent" context="scene">
<Scene camera="iso" height="100%">
  <Model id="frame" src={self.assets.frame}>
    <Move node="stage_x" axis="x" value={self.stage.x} gain={8} />
    <Move node="stage_y" axis="z" value={self.stage.y} gain={8} />
    <Glow node="light" on={self.stage.light_on || self.camera.live} color="warning" intensity={2} />
    <Popout trigger="hover" title="FRAME">
      <Text size="sm">At {self.stage.position} · z {fmt.number(value: self.stage.z, digits: 1)} µm · {self.stage.frames_acquired} frames</Text>
    </Popout>
  </Model>
</Scene>
</Blok>
"""

#: Panel: live view with overlays, and controls for live, exposure, stage, focus and snap.
#: Fills the page (``chrome="none"``).
FRAME_PANEL = """
<Blok kind="agent" context="panel" chrome="none">
  <local name="step" kind="STRING" default="100" />
  <local name="exposure" like="self.set_exposure.exposure_ms" />
  <div className="@container h-full">
  <div className="flex h-full flex-col gap-4 p-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex flex-col gap-0.5">
        <p className="text-base font-semibold tracking-tight">openUC2 FRAME</p>
        <p className="text-sm text-muted-foreground">{self.stage.position} · {self.stage.frames_acquired} frames</p>
      </div>
      <div className="flex items-center gap-2">
        <if when={self.camera.live}>
          <Badge variant="destructive" icon="circle-dot">live</Badge>
        </if>
        <Badge variant={self.stage.acquiring ? "default" : "outline"} icon={self.stage.acquiring ? "camera" : "check"}>{self.stage.acquiring ? "exposing" : "idle"}</Badge>
      </div>
    </div>

    <div className="flex min-h-0 flex-1 flex-col items-start gap-4 @4xl:flex-row">
      <div className="relative aspect-square w-full shrink-0 overflow-hidden rounded-xl bg-muted ring-1 ring-border @4xl:h-full @4xl:w-auto">
        <if when={self.camera.live}>
          <StructureDisplay value={list.first(value: self.camera.streams)} className="absolute inset-0 h-full w-full" emptyText="No stream" />
          <else>
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground">
              <Icon name="eye-off" size={32} />
              <p className="text-sm">Live off</p>
              <Button size="sm" icon="video" onClick={self.set_live(on: true)}>Start</Button>
            </div>
          </else>
        </if>
        <div className="pointer-events-none absolute left-[50%] top-[50%] -ml-px -mt-5 h-10 w-px bg-brand" />
        <div className="pointer-events-none absolute left-[50%] top-[50%] -ml-5 -mt-px h-px w-10 bg-brand" />
        <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-2 rounded-full bg-background/80 px-2.5 py-1 text-xs font-medium text-foreground shadow-sm">
          <span className={self.camera.streaming ? "size-2 rounded-full bg-destructive animate-pulse" : "size-2 rounded-full bg-muted-foreground"} />
          <span>{self.camera.streaming ? "LIVE" : "OFF"} · {fmt.number(value: self.camera.exposure_ms, digits: 0)} ms</span>
        </div>
        <div className="pointer-events-none absolute bottom-3 left-3 flex flex-col rounded-md bg-background/80 px-2 py-1 font-mono text-[11px] leading-tight text-foreground shadow-sm">
          <span>X {fmt.number(value: math.mul(a: self.stage.x, b: 1000), digits: 3)} mm</span>
          <span>Y {fmt.number(value: math.mul(a: self.stage.y, b: 1000), digits: 3)} mm</span>
          <span>Z {fmt.number(value: self.stage.z, digits: 1)} µm</span>
        </div>
        <div className="pointer-events-none absolute bottom-3 right-3 flex w-[50%] flex-col items-center gap-1 rounded-md bg-background/80 px-2 py-1 text-[11px] font-medium text-foreground shadow-sm">
          <div className="h-1 w-[89%] rounded-full bg-foreground" />
          <span>500 µm</span>
        </div>
      </div>

      <div className="flex w-full flex-col gap-3 @4xl:min-w-[18rem] @4xl:max-w-[28rem] @4xl:flex-1">
        <div className="flex flex-col gap-3 rounded-xl border bg-card p-3">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium">Live</p>
            <Switch checked={self.camera.live} onChange={self.set_live(on: event.value)} />
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Exposure</span>
              <span className="font-mono">{fmt.number(value: self.camera.exposure_ms, digits: 0)} ms</span>
            </div>
            <Slider bind={exposure} min={5} max={200} step={5} onCommit={self.set_exposure(exposure_ms: event.value)} />
          </div>
        </div>

        <div className="flex flex-col gap-3 rounded-xl border bg-card p-3">
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-medium">Stage</p>
            <ToggleGroup bind={step} size="sm" variant="outline">
              <ToggleGroupItem value="10" label="10 µm" />
              <ToggleGroupItem value="100" label="100 µm" />
              <ToggleGroupItem value="1000" label="1 mm" />
            </ToggleGroup>
          </div>
          <div className="flex flex-col items-center gap-1.5">
            <Button variant="outline" icon="arrow-up" onClick={self.jog(dx_um: 0, dy_um: math.neg(a: val.to_number(value: step) ?? 100))} />
            <div className="flex gap-1.5">
              <Button variant="outline" icon="arrow-left" onClick={self.jog(dx_um: math.neg(a: val.to_number(value: step) ?? 100), dy_um: 0)} />
              <Button variant="secondary" icon="home" disabled={self.stage.position == "home"} onClick={self.move_to_position(position: "home")} />
              <Button variant="outline" icon="arrow-right" onClick={self.jog(dx_um: val.to_number(value: step) ?? 100, dy_um: 0)} />
            </div>
            <Button variant="outline" icon="arrow-down" onClick={self.jog(dx_um: 0, dy_um: val.to_number(value: step) ?? 100)} />
          </div>
        </div>

        <div className="flex flex-col gap-3 rounded-xl border bg-card p-3">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium">Focus</p>
            <span className="font-mono text-xs text-muted-foreground">z {fmt.number(value: self.stage.z, digits: 1)} µm</span>
          </div>
          <div className="flex gap-1.5">
            <Button size="sm" variant="outline" icon="arrow-down" className="flex-1" onClick={self.focus(dz_um: -10)}>10</Button>
            <Button size="sm" variant="outline" icon="chevron-down" className="flex-1" onClick={self.focus(dz_um: -1)}>1</Button>
            <Button size="sm" variant="outline" icon="chevron-up" className="flex-1" onClick={self.focus(dz_um: 1)}>1</Button>
            <Button size="sm" variant="outline" icon="arrow-up" className="flex-1" onClick={self.focus(dz_um: 10)}>10</Button>
          </div>
          <Button size="sm" variant="secondary" icon="wand" fullWidth={true} disabled={self.tasks.run_autofocus.running} onClick={self.run_autofocus() into focused}>Autofocus</Button>
          <if when={focused.failed}>
            <Alert variant="destructive" icon="alert-triangle" title="Autofocus failed">{focused.error}</Alert>
          </if>
        </div>

        <Button icon="camera" fullWidth={true} disabled={self.tasks.snap.running} onClick={self.snap() into shot}>Snap</Button>
        <if when={self.tasks.snap.running}>
          <Progress value={self.tasks.snap.progress} max={100} />
        </if>
        <if when={!self.tasks.snap.running && self.camera.last_scene != null}>
          <Link href={str.concat(values: ["/mikro/scenes/", self.camera.last_scene.object])} newTab={true} className="flex items-center justify-center gap-2 rounded-md border bg-card px-3 py-2 text-sm font-medium no-underline hover:bg-muted">
            <Icon name="external-link" size={14} />
            <span>Open scene</span>
          </Link>
        </if>
        <if when={shot.failed}>
          <Alert variant="destructive" icon="alert-triangle" title="Snap failed">{shot.error}</Alert>
        </if>
      </div>
    </div>
  </div>
  </div>
</Blok>
"""

openuc_app.blok(
    "openuc-panel",
    FRAME_PANEL,
    description="Live view, stage, focus and snap.",
    default=True,
)

openuc_app.blok(
    "openuc-3d",
    FRAME_SCENE,
    description="The FRAME in 3D.",
    default=True,
)


#: Logo: three openUC2 cubes, light through a slide.
FRAME_LOGO = """
<Blok kind="agent" context="logo">
  <Logo viewBox="0 0 64 64" title="openUC2">
    <Defs>
      <LinearGradient id="frame-ground" x1="0" y1="0" x2="1" y2="1">
        <Stop offset="0" stopColor="ink" />
        <Stop offset="1" stopColor="brand-strong" />
      </LinearGradient>
      <RadialGradient id="frame-glow" cx="0.5" cy="0.45" r="0.6">
        <Stop offset="0" stopColor="spectrum-cyan" stopOpacity="0.45" />
        <Stop offset="1" stopColor="spectrum-cyan" stopOpacity="0" />
      </RadialGradient>
      <ClipPath id="frame-tile"><Rect x="0" y="0" width="64" height="64" rx="14" /></ClipPath>
    </Defs>
    <G clipPath="url(#frame-tile)">
      <Rect x="0" y="0" width="64" height="64" fill="url(#frame-ground)" />
      <Rect x="0" y="0" width="64" height="64" fill="url(#frame-glow)" />
      <G fill="surface" fillOpacity="0.12" stroke="surface" strokeOpacity="0.85" strokeWidth="1.5">
        <Rect x="21" y="7" width="22" height="14" rx="2.5" />
        <Rect x="21" y="25" width="22" height="14" rx="2.5" />
        <Rect x="21" y="43" width="22" height="14" rx="2.5" />
      </G>
      <Line x1="32" y1="14" x2="32" y2="50" stroke="spectrum-cyan" strokeWidth="2" strokeDasharray="3 2.5" animate="march" />
      <Circle cx="32" cy="13" r="3" fill="spectrum-cyan" />
      <Rect x="23.5" y="30.5" width="17" height="3" rx="0.8" fill="surface" />
      <Ellipse cx="28" cy="32" rx="1.6" ry="1.1" fill="spectrum-cyan" />
      <Ellipse cx="35.5" cy="32" rx="1.3" ry="1" fill="spectrum-cyan" fillOpacity="0.8" />
      <Path d="M26.5 46h11l-2.8 6.5h-5.4z" fill="surface" />
    </G>
    <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
  </Logo>
</Blok>
"""

openuc_app.blok(
    "openuc-logo",
    FRAME_LOGO,
    description="Logo.",
    default=True,
)

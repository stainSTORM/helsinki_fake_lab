"""helsinki-protocolapp: StainStorm for the Helsinki lab.

Its workflow ``process_slides`` moves slides from its own sample bay to the OT-2 and the
microscope and back: stain, image, stitch, segment, measure, restain while below target.
It knows the other apps only through the ``*Like`` protocols below. Layout and arm
trajectories come from testing/helsinki_layout.py.

Run:  uv run arkitekt run dev
      (or ../lab.py, which serves all seven apps together)
"""

from collections.abc import Generator
from dataclasses import field
from datetime import UTC, datetime
from typing import Annotated, Protocol

from arkitekt import ActionBlok, App, Description, Task, policy
from arkitekt.widgets import withStateChoices
from mikro.api.schema import TableLens
from mikro.arkitekt.specs import LabelMask, RGBImage
from rekuest.client.client import RekuestCaller

# =========================================================================== #
# helsinki-protocolapp: StainStorm, composing the six                          #
# =========================================================================== #
# ``process_slides``: stain, image, stitch, segment, measure; restain while below target, up
# to ``max_rounds``. No concurrent variant, on purpose: a replayed workflow cannot depend on
# which background task finished first.
IDENTIFIER = "helsinki-protocolapp"

lab_app = App(
    IDENTIFIER,
    "0.1.0",
    description=(
        "StainStorm for the Helsinki lab: stains, images, stitches, segments and measures "
        "slides, re-staining each until it reaches its target, by composing six other apps."
    ),
)
# lab.py --demo starts the run through this app's own client.
lab_app.register_client(RekuestCaller)

#: Sample bay: three slots in front of the arm (nodes ``slide_1`` to ``slide_3``). Written
#: by testing/helsinki_layout.py.
BAY_SLOTS = 3
lab_app.register_asset("bay", "models/sample_bay.glb")


# --------------------------------------------------------------------------- #
# What this app demands of the other six                                       #
# --------------------------------------------------------------------------- #
# Each field must mirror the provider's (key, kind, nullability) or the server refuses to
# assign; ``--check`` catches drift. Fields the workflow does not read are for the task
# panel.
class TrajectoriesLike(Protocol):
    """The robot's ``trajectories`` state."""

    available_trajectories: list[str]
    current_position: str
    gripping: bool


class RunLike(Protocol):
    """The liquid handler's ``run`` state."""

    available_protocols: list[str]
    current_protocol: str
    current_step: str
    runs_completed: int


class StageLike(Protocol):
    """The microscope's ``stage`` state."""

    position: str
    frames_acquired: int


@lab_app.declare(app="arkirino", auto_resolvable=True, min=1, max=1)
class ArkirinoLike(Protocol):
    """A robot arm with saved trajectories and a gripper."""

    trajectories: TrajectoriesLike

    def run_trajectory(
        self, name: str, speed: int | None = None, acceleration: int | None = None
    ) -> None:
        """Run Trajectory"""
        ...

    def grip(self) -> None:
        """Grip"""
        ...

    def ungrip(self) -> None:
        """Ungrip"""
        ...

    def retreat(self, speed: int | None = None) -> None:
        """Retreat"""
        ...


@lab_app.declare(app="ot2", auto_resolvable=True, min=1, max=1)
class OT2Like(Protocol):
    """A liquid handler that runs named protocols."""

    run: RunLike

    @policy(inquire_after="5s")
    def run_protocol(self, protocol: str) -> None:
        """Run Protocol"""
        ...


@lab_app.declare(app="openuc", auto_resolvable=True, min=1, max=1)
class FrameLike(Protocol):
    """A microscope with a stage."""

    stage: StageLike

    def move_to_position(self, position: str) -> None:
        """Move To Position"""
        ...

    def acquire_image(self) -> RGBImage:
        """Acquire Image"""
        ...


@lab_app.declare(app="stitcher", auto_resolvable=True, min=1)
class StitcherLike(Protocol):
    """An app that joins RGB tiles into one image."""

    def stitch_images(self, images: list[RGBImage]) -> RGBImage:
        """Stitch Images"""
        ...


@lab_app.declare(app="zellpose", auto_resolvable=True, min=1)
class SegmenterLike(Protocol):
    """An app that segments the cells of an RGB image: zellpose, Cellpose-SAM
    (``~/Code/apps/zellpose``), which runs on its own, outside this lab."""

    def segment_cells_rgb(
        self,
        image: RGBImage,
        diameter: float | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
    ) -> LabelMask:
        """Segment Cells in RGB Image"""
        ...


@lab_app.declare(app="analyzer", auto_resolvable=True, min=1)
class AnalyzerLike(Protocol):
    """An app that measures the cells of a slide: one row per cell, and the stained
    share of all cell pixels."""

    def measure_cells(
        self, image: RGBImage, labels: LabelMask
    ) -> tuple[TableLens, float]:
        """Measure Cells"""
        ...


# --------------------------------------------------------------------------- #
# What goes in, what comes out, and what is shown meanwhile                    #
# --------------------------------------------------------------------------- #
@lab_app.model
class Slide:
    """One loaded slide: which slot of the sample bay it sits in, and how it is to be stained."""

    name: Annotated[str, Description("What to call this slide")]
    protocol: Annotated[
        str,
        withStateChoices("opentrons.run.available_protocols"),
        Description("The protocol the liquid handler runs on this slide, every round"),
    ]
    slot: Annotated[
        int,
        Description(
            "The slot of the sample bay it sits in, 1 to 3 (0, or one already taken: the next free one)"
        ),
    ] = 0


#: Default slides: a full bay. Written as the wire value because port defaults go to the
#: server as JSON.
DEFAULT_SLIDES = [
    {"name": "slide A", "protocol": "dapi_stain", "slot": 1},
    {"name": "slide B", "protocol": "he_stain", "slot": 2},
    {"name": "slide C", "protocol": "wash_only", "slot": 3},
]


@lab_app.model
class SlideResult:
    """What one slide came to."""

    name: str
    stitched: RGBImage
    labels: LabelMask
    cells: TableLens
    stain: float
    rounds: int
    reached_target: bool


#: The steps a slide goes through, in order: (key in ``lab.phase``, label on the bloks).
PHASES: list[tuple[str, str]] = [
    ("pickup", "Pick up"),
    ("stain", "Stain"),
    ("image", "Image"),
    ("stitch", "Stitch"),
    ("segment", "Segment"),
    ("measure", "Measure"),
    ("return", "Put back"),
]

#: Where each step starts within one slide's share of the run's progress. A further
#: round goes through stain..measure again without moving the bar back.
PHASE_START = {
    "pickup": 0.0,
    "stain": 0.1,
    "image": 0.3,
    "stitch": 0.55,
    "segment": 0.7,
    "measure": 0.8,
    "return": 0.9,
}


@lab_app.state(name="lab")
class LabState:
    """Run progress per slide. The dicts are replaced whole on every change, never mutated.
    """

    phase: str = "idle"
    phase_index: int = -1
    current_slide: str = ""
    current_round: int = 0
    slide_progress: int = 0
    target_stain: float = 0.0
    slides: list[str] = field(default_factory=list)
    slide_status: dict[str, str] = field(default_factory=dict)
    slide_rounds: dict[str, int] = field(default_factory=dict)
    slide_stain: dict[str, float] = field(default_factory=dict)
    slides_done: int = 0
    #: The sample bay's slots holding their slide right now (1 to 3).
    bay: list[int] = field(default_factory=list)
    #: Last failure ("" if none). ``failed_index`` indexes PHASES, -1 if outside them.
    failure: str = ""
    failed_slide: str = ""
    failed_step: str = ""
    failed_index: int = -1
    failed_at: str = ""
    latest_stitched: RGBImage | None = None
    latest_labels: LabelMask | None = None
    latest_cells: TableLens | None = None


@lab_app.startup
def initialize_lab() -> LabState:
    """Start idle, with an empty board."""
    print(f"[{IDENTIFIER}] ready")
    return LabState()


# --------------------------------------------------------------------------- #
# Logos                                                                        #
# --------------------------------------------------------------------------- #
# Blok names are org-wide on the server, so each is prefixed with the identifier.
APP_LOGO = f"{IDENTIFIER}-logo"
PROCESS_LOGO = f"{IDENTIFIER}-process-slides-logo"

lab_app.blok(
    APP_LOGO,
    """
    <Blok kind="agent" context="logo">
    <Logo viewBox="0 0 64 64" title="Helsinki protocol app">
      <Defs>
        <LinearGradient id="ground" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0" stopColor="ink" />
          <Stop offset="1" stopColor="brand-strong" />
        </LinearGradient>
        <LinearGradient id="glass" x1="0" y1="0" x2="0" y2="1">
          <Stop offset="0" stopColor="surface" stopOpacity="0.9" />
          <Stop offset="1" stopColor="surface" stopOpacity="0.55" />
        </LinearGradient>
        <ClipPath id="tile"><Rect x="0" y="0" width="64" height="64" rx="14" /></ClipPath>
      </Defs>
      <G clipPath="url(#tile)">
        <Rect x="0" y="0" width="64" height="64" fill="url(#ground)" />
        <Rect x="26" y="2" width="12" height="4" rx="1.2" fill="muted" />
        <Path d="M27.5 6h9v9l-3 10h-3l-3-10z" fill="surface" fillOpacity="0.85" />
        <Path d="M32 27.5c-1.8 2.4-2.7 4-2.7 5.2a2.7 2.7 0 0 0 5.4 0c0-1.2-.9-2.8-2.7-5.2z" fill="brand" animate="float" />
        <Rect x="7" y="40" width="50" height="15" rx="2" fill="url(#glass)" stroke="surface" strokeOpacity="0.6" strokeWidth="0.8" />
        <Rect x="7.4" y="40.4" width="10" height="14.2" rx="1.6" fill="muted" fillOpacity="0.85" />
        <Ellipse cx="31" cy="47" rx="3.2" ry="2.4" fill="spectrum-blue" />
        <Ellipse cx="39" cy="45" rx="2.4" ry="1.9" fill="spectrum-violet" fillOpacity="0.85" />
        <Ellipse cx="45.5" cy="49" rx="2.8" ry="2.1" fill="spectrum-blue" fillOpacity="0.8" />
        <Ellipse cx="24.5" cy="50" rx="2" ry="1.6" fill="spectrum-violet" fillOpacity="0.7" />
        <Ellipse cx="51" cy="44.5" rx="1.6" ry="1.3" fill="spectrum-blue" fillOpacity="0.6" />
      </G>
      <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
    </Logo>
    </Blok>
    """,
    description="A slide whose cells take up the stain dropping from a pipette tip.",
    default=True,
)

lab_app.blok(
    PROCESS_LOGO,
    """
    <Blok kind="action" context="logo">
    <Logo viewBox="0 0 24 24" title="Stain, image and measure slides until they are stained enough">
      <G fill="none" stroke="ink" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <Rect x="2.5" y="14.5" width="19" height="7" rx="1.5" fill="surface" />
        <Circle cx="11" cy="18" r="1.6" fill="brand" stroke="none" animate="pulse" />
        <Circle cx="15.5" cy="17.6" r="1.1" fill="brand" stroke="none" />
        <Path d="M6.5 9.5a5.5 5.5 0 0 1 10.3-2.7" />
        <Path d="M17.5 3.6v3.6h-3.6" />
        <Path d="M17.5 9.5a5.5 5.5 0 0 1-10.3 2.7" />
        <Path d="M12 7.2c-.9 1.2-1.4 2-1.4 2.6a1.4 1.4 0 0 0 2.8 0c0-.6-.5-1.4-1.4-2.6z" fill="brand-soft" strokeWidth="1.2" />
      </G>
    </Logo>
    </Blok>
    """,
    description="The workflow's logo: a slide, and the stain going round until it takes.",
)


# --------------------------------------------------------------------------- #
# Bloks                                                                        #
# --------------------------------------------------------------------------- #
# Plain BSX, no f-strings, so the language server checks them as written. Both panels
# draw the steps from a literal list whose `i` is the step's index in PHASES.

#: Task panel: the lab in 3D on the left, run status on the right. Same room as LAB below.
TASK_PANEL = """
<Blok kind="action" context="panel" chrome="none">
  <div className="@container h-full">
    <div className="flex h-full flex-col gap-4 @4xl:flex-row">
      <div className="relative h-[32rem] min-w-0 shrink-0 overflow-hidden rounded-2xl border bg-linear-to-b from-muted/50 to-background shadow-sm @4xl:h-auto @4xl:flex-1">
          <local name="show_path" default={true} persist />
        <Space room="room:helsinki-lab" camera="iso">
          <Placed dep="robot" matrix={[[0, 0, 1, 0], [0, 1, 0, 0.0101], [-1, 0, 0, 0], [0, 0, 0, 1]]}>
            <Glow on={self.lab.phase == "pickup" || self.lab.phase == "return"} color="warning" />
            <Popout open={self.lab.phase == "pickup" || self.lab.phase == "return"} title={self.lab.current_slide}>
              <p className="text-sm">{deps.robot.trajectories.gripping ? "Carrying the slide" : "Moving"}, at <span className="font-mono text-xs">{deps.robot.trajectories.current_position}</span></p>
            </Popout>
          </Placed>
          <Placed dep="opentrons" matrix={[[1, 0, 0, -0.78], [0, 1, 0, 0.015], [0, 0, 1, 0], [0, 0, 0, 1]]}>
            <Glow on={self.lab.phase == "stain"} color="brand" />
            <Popout open={self.lab.phase == "stain"} title={self.lab.current_slide}>
              <p className="text-sm">Round {self.lab.current_round}: {deps.opentrons.run.current_step}</p>
            </Popout>
          </Placed>
          <Placed dep="frame" matrix={[[0, 0, -1, 0.56], [0, 1, 0, 0.016], [1, 0, 0, 0], [0, 0, 0, 1]]}>
            <Glow on={self.lab.phase == "image"} color="info" />
            <Popout open={self.lab.phase == "image"} title={self.lab.current_slide}>
              <p className="text-sm">At <span className="font-mono text-xs">{deps.frame.stage.position}</span></p>
            </Popout>
          </Placed>
          <if when={show_path}>
            <Route through={["robot", "opentrons", "frame", "robot"]} progress={self.lab.slide_progress} color="chart-2" />
          </if>
          <Transform matrix={[[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0.4], [0, 0, 0, 1]]}>
            <Model id="bay" src={self.assets.bay}>
              <Visible node="slide_1" on={list.contains(value: self.lab.bay, item: 1)} />
              <Visible node="slide_2" on={list.contains(value: self.lab.bay, item: 2)} />
              <Visible node="slide_3" on={list.contains(value: self.lab.bay, item: 3)} />
            </Model>
          </Transform>
        </Space>
        <div className="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2.5 rounded-full border bg-background/90 py-1.5 pr-4 pl-3 shadow-sm">
            <span className="relative flex size-2 shrink-0">
              <span className={["absolute inline-flex size-full rounded-full opacity-70", task.running && "animate-ping", task.failed ? "bg-destructive" : task.done ? "bg-success" : "bg-brand"]} />
              <span className={["relative inline-flex size-2 rounded-full", task.failed ? "bg-destructive" : task.done ? "bg-success" : "bg-brand"]} />
            </span>
            <span className="shrink-0 text-sm font-semibold tracking-tight">{str.length(value: self.lab.current_slide) > 0 ? self.lab.current_slide : "Process slides"}</span>
            <span className="truncate text-xs text-muted-foreground">{task.message ?? "Waiting to start"}</span>
          </div>
          <span className={["shrink-0 rounded-full border px-3 py-1.5 text-xs font-medium uppercase tracking-wider shadow-sm", task.failed ? "border-destructive/40 bg-destructive/10 text-destructive" : task.done ? "border-success/40 bg-success-soft text-success" : task.running ? "border-brand/40 bg-brand-soft text-brand" : "bg-background/90 text-muted-foreground"]}>{task.status}</span>
        </div>
        <div className="absolute inset-x-3 bottom-3 rounded-xl border bg-background/90 px-4 py-3 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-semibold tabular-nums tracking-tight">{task.progress}%</span>
              <span className="text-xs text-muted-foreground">{self.lab.slides_done} of {list.len(value: args.loaded_slides)} slides done</span>
              <if when={self.lab.current_round > 0}>
                <span className="text-xs text-muted-foreground">· round {self.lab.current_round} of {args.max_rounds}</span>
              </if>
              <if when={task.duration != null}>
                <span className="text-xs text-muted-foreground">· took {fmt.duration(seconds: task.duration)}</span>
                <else>
                  <if when={task.started_at != null}>
                    <span className="text-xs text-muted-foreground">· started <Time value={task.started_at} relative={true} /></span>
                  </if>
                </else>
              </if>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              <Switch bind={show_path} label="Slide's route" />
              <if when={task.can.pause}>
                <Button variant="outline" size="sm" icon="pause" onClick={task.pause()}>Pause</Button>
              </if>
              <if when={task.can.resume}>
                <Button size="sm" icon="play" onClick={task.resume()}>Resume</Button>
              </if>
              <if when={task.can.cancel}>
                <Button variant="outline" size="sm" icon="x" onClick={task.cancel()}>Cancel</Button>
              </if>
            </div>
          </div>
          <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-muted">
            <div className={["h-full rounded-full transition-all duration-700 ease-out", task.failed ? "bg-destructive" : task.paused ? "bg-warning" : "bg-linear-to-r from-brand to-info"]} width={str.concat(values: [task.progress, "%"])} />
          </div>
          <timeline />
        </div>
      </div>
      <div className="flex min-w-0 flex-col gap-3 @4xl:min-h-0 @4xl:w-[24rem] @4xl:shrink-0 @4xl:overflow-y-auto">
        <ask />
        <if when={task.failed}>
          <div className="flex gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-destructive">
            <Icon name="alert-triangle" size={16} className="mt-0.5 shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-medium">Failed at {self.lab.failed_step}{str.length(value: self.lab.failed_slide) > 0 ? str.concat(values: [" · ", self.lab.failed_slide]) : ""}</p>
              <p className="whitespace-pre-wrap break-words font-mono text-xs opacity-80">{task.error}</p>
            </div>
          </div>
        </if>
        <if when={task.paused}>
          <div className="flex items-center gap-3 rounded-xl border border-warning/40 bg-warning-soft p-3 text-warning">
            <Icon name="pause" size={16} className="shrink-0" />
            <p className="text-xs">Paused: the devices hold where they are until someone resumes the run.</p>
          </div>
        </if>
        <if when={self.lab.current_round > 1}>
          <div className="flex items-center gap-3 rounded-xl border border-info/40 bg-info-soft p-3 text-info">
            <Icon name="refresh" size={16} className="shrink-0" />
            <p className="text-xs">Staining again: the last measure was below {fmt.percent(value: args.target_stain, digits: 0)}.</p>
          </div>
        </if>
        <div className="rounded-2xl border bg-card p-4 shadow-sm">
          <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">What is happening</p>
          <div className="relative">
            <div className="absolute top-3 bottom-3 left-3 w-px bg-border" />
            <foreach items={[{i: 0, label: "Pick up", by: "Arm"}, {i: 1, label: "Stain", by: "Liquid handler"}, {i: 2, label: "Image", by: "Microscope"}, {i: 3, label: "Stitch", by: "Stitcher"}, {i: 4, label: "Segment", by: "Segmenter"}, {i: 5, label: "Measure", by: "Analyzer"}, {i: 6, label: "Put back", by: "Arm"}]} let="step" key={step.i}>
              <div className="relative flex items-center gap-3 pb-2.5 last:pb-0">
                <div className={["z-10 flex size-6 shrink-0 items-center justify-center rounded-full border transition-all duration-300", self.lab.phase_index > step.i ? "border-success/40 bg-success-soft text-success" : self.lab.phase_index == step.i ? "border-brand bg-brand text-primary-foreground shadow-md shadow-brand/30 ring-4 ring-brand/15" : "border-border bg-card text-muted-foreground"]}>
                  <Icon name={self.lab.phase_index > step.i ? "check" : self.lab.phase_index == step.i ? "loader" : "circle"} size={12} className={[self.lab.phase_index == step.i && "animate-spin", self.lab.phase_index < step.i && "opacity-40"]} />
                </div>
                <span className={["text-sm transition-colors", self.lab.phase_index == step.i ? "font-semibold" : self.lab.phase_index > step.i ? "text-foreground" : "text-muted-foreground"]}>{step.label}</span>
                <span className={["ml-auto rounded-full px-2 py-0.5 text-[11px] transition-colors", self.lab.phase_index == step.i ? "bg-brand-soft font-medium text-brand" : "text-muted-foreground"]}>{step.by}</span>
              </div>
            </foreach>
          </div>
        </div>
        <div className="grid gap-2">
          <div className={["flex items-center gap-3 rounded-xl border p-2.5 transition-all duration-300", (self.lab.phase == "pickup" || self.lab.phase == "return") ? "border-warning/50 bg-warning-soft shadow-sm" : "bg-card"]}>
            <div className={["flex size-9 shrink-0 items-center justify-center rounded-lg transition-colors", (self.lab.phase == "pickup" || self.lab.phase == "return") ? "bg-warning text-warning-foreground" : "bg-muted text-muted-foreground"]}>
              <Icon name="move" size={16} />
            </div>
            <div className="min-w-0 flex-1">
              <p className="flex items-center gap-1.5 text-sm leading-tight font-medium">Arm<span className={["size-1.5 shrink-0 rounded-full", deps.robot.connected ? "bg-success" : "bg-destructive"]} /></p>
              <p className="truncate text-xs text-muted-foreground">at {deps.robot.trajectories.current_position}</p>
            </div>
            <span className={["shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-medium", deps.robot.trajectories.gripping ? "border-warning/40 bg-warning text-warning-foreground" : "text-muted-foreground"]}>{deps.robot.trajectories.gripping ? "holding a slide" : "gripper open"}</span>
          </div>
          <div className={["flex items-center gap-3 rounded-xl border p-2.5 transition-all duration-300", self.lab.phase == "stain" ? "border-brand/50 bg-brand-soft shadow-sm" : "bg-card"]}>
            <div className={["flex size-9 shrink-0 items-center justify-center rounded-lg transition-colors", self.lab.phase == "stain" ? "bg-brand text-primary-foreground" : "bg-muted text-muted-foreground"]}>
              <Icon name="activity" size={16} />
            </div>
            <div className="min-w-0 flex-1">
              <p className="flex items-center gap-1.5 text-sm leading-tight font-medium">Liquid handler<span className={["size-1.5 shrink-0 rounded-full", deps.opentrons.connected ? "bg-success" : "bg-destructive"]} /></p>
              <p className="truncate text-xs text-muted-foreground">{str.length(value: deps.opentrons.run.current_step) > 0 ? deps.opentrons.run.current_step : "no step running"}</p>
            </div>
            <span className="shrink-0 rounded-full border px-2 py-0.5 font-mono text-[11px] text-muted-foreground">{str.length(value: deps.opentrons.run.current_protocol) > 0 ? deps.opentrons.run.current_protocol : "idle"}</span>
          </div>
          <div className={["flex items-center gap-3 rounded-xl border p-2.5 transition-all duration-300", self.lab.phase == "image" ? "border-info/50 bg-info-soft shadow-sm" : "bg-card"]}>
            <div className={["flex size-9 shrink-0 items-center justify-center rounded-lg transition-colors", self.lab.phase == "image" ? "bg-info text-info-foreground" : "bg-muted text-muted-foreground"]}>
              <Icon name="microscope" size={16} />
            </div>
            <div className="min-w-0 flex-1">
              <p className="flex items-center gap-1.5 text-sm leading-tight font-medium">Microscope<span className={["size-1.5 shrink-0 rounded-full", deps.frame.connected ? "bg-success" : "bg-destructive"]} /></p>
              <p className="truncate text-xs text-muted-foreground">stage at {deps.frame.stage.position}</p>
            </div>
            <span className="shrink-0 rounded-full border px-2 py-0.5 text-[11px] text-muted-foreground tabular-nums">{deps.frame.stage.frames_acquired} frames</span>
          </div>
        </div>
        <div className="rounded-2xl border bg-card p-3 shadow-sm">
          <Tabs defaultValue="slides">
            <TabsList className="w-full">
              <TabsTrigger value="slides" label="Slides" />
              <TabsTrigger value="mosaic" label="Mosaic" />
              <TabsTrigger value="labels" label="Labels" />
              <TabsTrigger value="results" label="Results" />
              <TabsTrigger value="log" label="Log" />
            </TabsList>
            <TabsContent value="slides">
              <div className="grid gap-2 pt-1">
                <foreach items={args.loaded_slides} let="s" key={s.name}>
                  <div className={["rounded-xl border p-2.5 transition-colors duration-300", s.name == self.lab.current_slide ? "border-brand/40 bg-brand-soft/50" : "bg-background/50"]}>
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium">{s.name}</span>
                      <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">{s.protocol}</span>
                      <div className="ml-auto shrink-0">
                        <match value={obj.get(value: self.lab.slide_status, key: s.name, default: "queued")}>
                          <case is="done"><span className="rounded-full border border-success/40 bg-success-soft px-2 py-0.5 text-[11px] font-medium text-success">done</span></case>
                          <case is="below target"><span className="rounded-full border border-warning/40 bg-warning-soft px-2 py-0.5 text-[11px] font-medium text-warning">below target</span></case>
                          <case is="restaining"><span className="animate-pulse rounded-full border border-info/40 bg-info-soft px-2 py-0.5 text-[11px] font-medium text-info">staining again</span></case>
                          <case is="running"><span className="animate-pulse rounded-full border border-brand/40 bg-brand-soft px-2 py-0.5 text-[11px] font-medium text-brand">in progress</span></case>
                          <default><span className="rounded-full border px-2 py-0.5 text-[11px] text-muted-foreground">queued</span></default>
                        </match>
                      </div>
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                        <div className={["h-full rounded-full transition-all duration-700 ease-out", obj.get(value: self.lab.slide_stain, key: s.name, default: 0) >= args.target_stain ? "bg-success" : "bg-linear-to-r from-spectrum-violet to-spectrum-blue"]} width={str.concat(values: [math.round(value: math.clamp(value: math.mul(a: math.div(a: obj.get(value: self.lab.slide_stain, key: s.name, default: 0), b: args.target_stain), b: 100), min: 0, max: 100), digits: 0), "%"])} />
                      </div>
                      <span className="w-12 text-right text-xs font-medium tabular-nums">{obj.has(value: self.lab.slide_stain, key: s.name) ? fmt.percent(value: obj.get(value: self.lab.slide_stain, key: s.name), digits: 1) : "–"}</span>
                      <span className="w-14 text-right text-[11px] text-muted-foreground tabular-nums">{obj.get(value: self.lab.slide_rounds, key: s.name, default: 0)} rounds</span>
                    </div>
                  </div>
                </foreach>
                <p className="text-[11px] text-muted-foreground">A full bar is the target of {fmt.percent(value: args.target_stain, digits: 0)} stained.</p>
              </div>
            </TabsContent>
            <TabsContent value="mosaic">
              <div className="overflow-hidden rounded-xl border">
                <StructureViewer value={self.lab.latest_stitched} height="16rem" emptyText="The first mosaic shows up here." />
              </div>
            </TabsContent>
            <TabsContent value="labels">
              <div className="overflow-hidden rounded-xl border">
                <StructureViewer value={self.lab.latest_labels} height="16rem" emptyText="The labels show up here." />
              </div>
            </TabsContent>
            <TabsContent value="results">
              <div className="grid gap-2 pt-1">
                <returns />
                <if when={self.lab.slides_done == 0}>
                  <p className="text-[11px] text-muted-foreground">Each slide's mosaic, labels and stain show up here as it is put back.</p>
                </if>
              </div>
            </TabsContent>
            <TabsContent value="log">
              <if when={list.len(value: task.log) > 0}>
                <List height="16rem">
                  <foreach items={list.reverse(value: task.log)} let="entry">
                    <div className="flex items-baseline gap-2 border-b py-1.5 last:border-b-0">
                      <span className="size-1.5 shrink-0 rounded-full bg-muted-foreground/40" />
                      <span className="min-w-0 flex-1 text-xs">{entry.message ?? entry.kind}</span>
                      <Time value={entry.at} relative={true} className="shrink-0 text-[11px] text-muted-foreground tabular-nums" />
                    </div>
                  </foreach>
                </List>
                <else>
                  <Empty icon="list" title="Nothing reported yet" description="What the run reports shows up here, newest first." />
                </else>
              </if>
            </TabsContent>
          </Tabs>
        </div>
        <Toast task={task}>
          <done>Every slide is back in its slot</done>
          <failed>The run failed: {task.error}</failed>
          <cancelled>The run was cancelled</cancelled>
        </Toast>
      </div>
    </div>
  </div>
</Blok>
"""

#: The lab in 3D. The room ("room:helsinki-lab") is shared by the whole organization. The
#: matrices are the default placements (rows, metres, Y up, table top at 0), printed by
#: helsinki_layout.py. The arm's trajectories are solved for these defaults, so it ignores
#: devices moved in the room. Devices are named by dependency key.
LAB = """
<Blok kind="action" context="scene" chrome="none">
<local name="show_path" default={true} persist />
<Space room="room:helsinki-lab" camera="iso">
  <Placed dep="robot" matrix={[[0, 0, 1, 0], [0, 1, 0, 0.0101], [-1, 0, 0, 0], [0, 0, 0, 1]]}>
    <Glow on={self.lab.phase == "pickup" || self.lab.phase == "return"} color="warning" />
    <Popout open={task.ask != null} title="The run asks">
      <ask />
    </Popout>
    <Popout trigger="click" title="This run">
      <Text size="sm">{self.lab.slides_done} of {list.len(value: args.loaded_slides)} slides done</Text>
      <Switch bind={show_path} label="Draw the slide's route" />
    </Popout>
  </Placed>
  <Placed dep="opentrons" matrix={[[1, 0, 0, -0.78], [0, 1, 0, 0.015], [0, 0, 1, 0], [0, 0, 0, 1]]}>
    <Glow on={self.lab.phase == "stain"} color="brand" />
    <Popout open={self.lab.phase == "stain"} title={self.lab.current_slide}>
      <Text size="sm">Round {self.lab.current_round}: {deps.opentrons.run.current_step}</Text>
    </Popout>
  </Placed>
  <Placed dep="frame" matrix={[[0, 0, -1, 0.56], [0, 1, 0, 0.016], [1, 0, 0, 0], [0, 0, 0, 1]]}>
    <Glow on={self.lab.phase == "image"} color="info" />
    <Popout open={self.lab.phase == "image"} title={self.lab.current_slide}>
      <Text size="sm">At {deps.frame.stage.position}</Text>
    </Popout>
  </Placed>
  <if when={show_path}>
    <Route through={["robot", "opentrons", "frame", "robot"]} progress={self.lab.slide_progress} color="chart-2" />
  </if>
  <Transform matrix={[[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0.4], [0, 0, 0, 1]]}>
    <Model id="bay" src={self.assets.bay}>
      <Visible node="slide_1" on={list.contains(value: self.lab.bay, item: 1)} />
      <Visible node="slide_2" on={list.contains(value: self.lab.bay, item: 2)} />
      <Visible node="slide_3" on={list.contains(value: self.lab.bay, item: 3)} />
    </Model>
  </Transform>
</Space>
</Blok>
"""


#: Assign blok: the lab on the left, readiness and the argument form on the right. No task
#: exists yet: ``args`` are raw form values (every read needs a fallback), ``deps`` the
#: agents a run would get. A held slide or no slides blocks the start, but only from here.
PREPARE = """
<Blok kind="assign" context="panel" chrome="none">
  <div className="@container h-full">
    <div className="flex h-full flex-col gap-4 @4xl:flex-row">
      <div className="relative h-[32rem] min-w-0 shrink-0 overflow-hidden rounded-2xl border bg-linear-to-b from-muted/50 to-background shadow-sm @4xl:h-auto @4xl:flex-1">
        <Space room="room:helsinki-lab" camera="iso">
          <Placed dep="robot" matrix={[[0, 0, 1, 0], [0, 1, 0, 0.0101], [-1, 0, 0, 0], [0, 0, 0, 1]]}>
            <Glow on={homing.running || releasing.running || deps.robot.trajectories.gripping} color="warning" />
            <Popout trigger="hover" title="Arm">
              <p className="text-sm">At <span className="font-mono text-xs">{deps.robot.trajectories.current_position}</span>, {deps.robot.trajectories.gripping ? "holding a slide" : "gripper open"}</p>
            </Popout>
          </Placed>
          <Placed dep="opentrons" matrix={[[1, 0, 0, -0.78], [0, 1, 0, 0.015], [0, 0, 1, 0], [0, 0, 0, 1]]}>
            <Popout trigger="hover" title="Liquid handler">
              <p className="font-mono text-xs">{list.join(value: deps.opentrons.run.available_protocols, separator: ", ")}</p>
            </Popout>
          </Placed>
          <Placed dep="frame" matrix={[[0, 0, -1, 0.56], [0, 1, 0, 0.016], [1, 0, 0, 0], [0, 0, 0, 1]]}>
            <Popout trigger="hover" title="Microscope">
              <p className="text-sm">Stage at <span className="font-mono text-xs">{deps.frame.stage.position}</span>, {deps.frame.stage.frames_acquired} frames taken</p>
            </Popout>
          </Placed>
          <Transform matrix={[[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0.4], [0, 0, 0, 1]]}>
            <Model id="bay" src={self.assets.bay}>
              <Visible node="slide_1" on={list.contains(value: list.pluck(value: args.loaded_slides ?? [], key: "slot"), item: 1)} />
              <Visible node="slide_2" on={list.contains(value: list.pluck(value: args.loaded_slides ?? [], key: "slot"), item: 2)} />
              <Visible node="slide_3" on={list.contains(value: list.pluck(value: args.loaded_slides ?? [], key: "slot"), item: 3)} />
            </Model>
          </Transform>
        </Space>
        <div className="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between gap-2">
          <div className="flex items-center gap-2.5 rounded-full border bg-background/90 py-1.5 pr-4 pl-3 shadow-sm">
            <span className="relative flex size-2 shrink-0">
              <span className={["absolute inline-flex size-full animate-ping rounded-full opacity-70", deps.robot.trajectories.gripping ? "bg-warning" : "bg-success"]} />
              <span className={["relative inline-flex size-2 rounded-full", deps.robot.trajectories.gripping ? "bg-warning" : "bg-success"]} />
            </span>
            <span className="text-sm font-semibold tracking-tight">The lab as it stands</span>
            <span className="text-xs text-muted-foreground">hover a device</span>
          </div>
          <span className={["shrink-0 rounded-full border px-3 py-1.5 text-xs font-medium uppercase tracking-wider shadow-sm", deps.robot.trajectories.gripping ? "border-warning/40 bg-warning-soft text-warning" : "border-success/40 bg-success-soft text-success"]}>{deps.robot.trajectories.gripping ? "needs a hand" : "ready"}</span>
        </div>
        <div className="pointer-events-none absolute inset-x-3 bottom-3 flex flex-wrap items-center gap-1.5 rounded-xl border bg-background/90 px-3 py-2.5 shadow-sm">
          <foreach items={[{i: 0, label: "Pick up"}, {i: 1, label: "Stain"}, {i: 2, label: "Image"}, {i: 3, label: "Stitch"}, {i: 4, label: "Segment"}, {i: 5, label: "Measure"}, {i: 6, label: "Put back"}]} let="step" key={step.i}>
            <if when={step.i > 0}>
              <Icon name="chevron-right" size={12} className="text-muted-foreground/60" />
            </if>
            <span className={["rounded-full border px-2.5 py-0.5 text-[11px] font-medium", (step.i > 0 && step.i < 6) ? "border-brand/30 bg-brand-soft text-brand" : "bg-background text-muted-foreground"]}>{step.label}</span>
          </foreach>
          <span className="ml-auto flex items-center gap-1 text-[11px] text-muted-foreground">
            <Icon name="refresh" size={11} />
            Stain to Measure repeats while below target
          </span>
        </div>
      </div>
      <div className="flex min-w-0 flex-col gap-3 @4xl:min-h-0 @4xl:w-[26rem] @4xl:shrink-0 @4xl:overflow-y-auto">
        <div className="rounded-2xl border bg-card p-4 shadow-sm">
          <div className="mb-3 flex items-baseline justify-between gap-2">
            <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">The devices this run gets</p>
            <p className="text-[11px] text-muted-foreground">what you prepare is what runs</p>
          </div>
          <div className="grid gap-2">
            <div className={["flex items-center gap-3 rounded-xl border p-2.5 transition-all duration-300", deps.robot.trajectories.gripping ? "border-warning/50 bg-warning-soft" : "bg-background/50"]}>
              <div className={["flex size-9 shrink-0 items-center justify-center rounded-lg transition-colors", deps.robot.trajectories.gripping ? "bg-warning text-warning-foreground" : "bg-muted text-muted-foreground"]}>
                <Icon name="move" size={16} className={[homing.running && "animate-pulse"]} />
              </div>
              <div className="min-w-0 flex-1">
                <p className="flex items-center gap-1.5 text-sm leading-tight font-medium">Arm<span className={["size-1.5 shrink-0 rounded-full", deps.robot.connected ? "bg-success" : "bg-destructive"]} /></p>
                <p className="truncate text-xs text-muted-foreground">at {deps.robot.trajectories.current_position}, {deps.robot.trajectories.gripping ? "holding a slide" : "gripper open"}</p>
              </div>
              <if when={deps.robot.trajectories.gripping}>
                <Button variant="outline" size="sm" icon="x" disabled={releasing.running} onClick={deps.robot.ungrip() into releasing}>Release</Button>
              </if>
              <Button variant="outline" size="sm" icon="home" disabled={homing.running || deps.robot.trajectories.current_position == "home"} onClick={deps.robot.run_trajectory(name: "home") into homing}>{deps.robot.trajectories.current_position == "home" ? "Home" : "Send home"}</Button>
            </div>
            <div className="flex items-center gap-3 rounded-xl border bg-background/50 p-2.5">
              <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                <Icon name="activity" size={16} />
              </div>
              <div className="min-w-0 flex-1">
                <p className="flex items-center gap-1.5 text-sm leading-tight font-medium">Liquid handler<span className={["size-1.5 shrink-0 rounded-full", deps.opentrons.connected ? "bg-success" : "bg-destructive"]} /></p>
                <p className="truncate text-xs text-muted-foreground">{list.len(value: deps.opentrons.run.available_protocols)} protocols, {deps.opentrons.run.runs_completed} runs so far</p>
              </div>
              <span className="shrink-0 rounded-full border px-2 py-0.5 font-mono text-[11px] text-muted-foreground">{str.length(value: deps.opentrons.run.current_protocol) > 0 ? deps.opentrons.run.current_protocol : "idle"}</span>
            </div>
            <div className="flex items-center gap-3 rounded-xl border bg-background/50 p-2.5">
              <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                <Icon name="microscope" size={16} />
              </div>
              <div className="min-w-0 flex-1">
                <p className="flex items-center gap-1.5 text-sm leading-tight font-medium">Microscope<span className={["size-1.5 shrink-0 rounded-full", deps.frame.connected ? "bg-success" : "bg-destructive"]} /></p>
                <p className="truncate text-xs text-muted-foreground">stage at {deps.frame.stage.position}</p>
              </div>
              <span className="shrink-0 rounded-full border px-2 py-0.5 text-[11px] text-muted-foreground tabular-nums">{deps.frame.stage.frames_acquired} frames</span>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-muted-foreground">then</span>
            <span className="flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium"><Icon name="grid" size={11} />{deps.stitcher.name}</span>
            <span className="flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium"><Icon name="layers" size={11} />{deps.segmenter.name}</span>
            <span className="flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium"><Icon name="gauge" size={11} />{deps.analyzer.name}</span>
          </div>
          <Accordion collapsible={true} className="mt-2">
            <AccordionItem value="who">
              <AccordionTrigger icon="users" label="Choose who runs each part" />
              <AccordionContent>
                <deps />
              </AccordionContent>
            </AccordionItem>
          </Accordion>
          <if when={homing.failed}>
            <div className="mt-3 flex gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-destructive">
              <Icon name="alert-triangle" size={16} className="mt-0.5 shrink-0" />
              <div className="min-w-0"><p className="text-sm font-medium">The arm did not get home</p><p className="text-xs opacity-80">{homing.error}</p></div>
            </div>
          </if>
          <if when={releasing.failed}>
            <div className="mt-3 flex gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-destructive">
              <Icon name="alert-triangle" size={16} className="mt-0.5 shrink-0" />
              <div className="min-w-0"><p className="text-sm font-medium">The gripper did not open</p><p className="text-xs opacity-80">{releasing.error}</p></div>
            </div>
          </if>
        </div>
        <div className="rounded-2xl border bg-card p-4 shadow-sm">
          <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">The run</p>
          <Popover>
            <PopoverTrigger className="group block w-full cursor-pointer rounded-xl border bg-background/60 p-3 text-left transition-colors hover:border-brand/40 hover:bg-brand-soft/30">
              <div className="flex items-center gap-2">
                <Icon name="layers" size={14} className="shrink-0 text-brand" />
                <span className="text-sm font-medium">Slide holder</span>
                <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] tabular-nums text-muted-foreground">{list.len(value: args.loaded_slides ?? [])} loaded</span>
                <span className="ml-auto flex items-center gap-1 text-[11px] text-muted-foreground transition-colors group-hover:text-brand">Edit<Icon name="chevron-right" size={12} /></span>
              </div>
              <if when={list.len(value: args.loaded_slides ?? []) > 0}>
                <Grid columns={2} gap="0.375rem" margin="0.625rem 0 0 0">
                  <foreach items={args.loaded_slides ?? []} let="s">
                    <div className="flex min-w-0 flex-col rounded-lg border bg-card px-2.5 py-1.5 shadow-xs">
                      <span className="truncate text-sm font-medium">{s.name ?? "unnamed"}</span>
                      <span className={["truncate font-mono text-[10px]", s.protocol == null || s.slot == null ? "text-destructive" : "text-muted-foreground"]}>{s.protocol ?? "no protocol"} · {s.slot != null ? str.concat(values: ["slot ", s.slot]) : "no slot"}</span>
                    </div>
                  </foreach>
                </Grid>
                <else>
                  <div className="mt-2.5 flex items-center justify-center gap-2 rounded-lg border border-dashed py-4 text-sm text-muted-foreground">
                    <Icon name="plus" size={14} />Load slides
                  </div>
                </else>
              </if>
            </PopoverTrigger>
            <PopoverContent side="left" align="start" sideOffset={12} className="w-[26rem] gap-3 p-4">
              <div>
                <p className="text-sm font-semibold">Slide holder</p>
                <p className="text-xs text-muted-foreground">Each slide in the order it is run: its name, its protocol and its bay slot.</p>
              </div>
              <arg key="loaded_slides" />
            </PopoverContent>
          </Popover>
          <Accordion collapsible={true}>
            <AccordionItem value="tuning">
              <AccordionTrigger icon="sliders" label="Trajectories, tiles and target" />
              <AccordionContent>
                <args />
              </AccordionContent>
            </AccordionItem>
          </Accordion>
          <Grid columns={3} gap="0.5rem" margin="1rem 0 0 0">
            <div className="rounded-xl border bg-linear-to-br from-brand-soft to-transparent p-3">
              <p className="text-2xl leading-none font-semibold tabular-nums tracking-tight">{list.len(value: args.loaded_slides ?? [])}</p>
              <p className="mt-1 text-[11px] text-muted-foreground">slides</p>
            </div>
            <div className="rounded-xl border bg-linear-to-br from-info-soft to-transparent p-3">
              <p className="text-2xl leading-none font-semibold tabular-nums tracking-tight">{fmt.number(value: math.mul(a: list.len(value: args.loaded_slides ?? []), b: args.tiles ?? 3), digits: 0)}</p>
              <p className="mt-1 text-[11px] text-muted-foreground">tiles per round</p>
            </div>
            <div className="rounded-xl border bg-linear-to-br from-success-soft to-transparent p-3">
              <p className="text-2xl leading-none font-semibold tabular-nums tracking-tight">{fmt.percent(value: args.target_stain ?? 0.1, digits: 0)}</p>
              <p className="mt-1 text-[11px] text-muted-foreground">in {args.max_rounds ?? 3} rounds at most</p>
            </div>
          </Grid>
          <if when={deps.robot.trajectories.gripping}>
            <div className="mt-3 flex gap-3 rounded-xl border border-warning/40 bg-warning-soft p-3 text-warning">
              <Icon name="alert-triangle" size={16} className="mt-0.5 shrink-0" />
              <div className="min-w-0"><p className="text-sm font-medium">The arm is holding a slide</p><p className="text-xs opacity-80">Release it before a run: the first thing a run does is pick one up.</p></div>
            </div>
          </if>
          <if when={assign.blocked != null}>
            <div className="mt-3 flex gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-destructive">
              <Icon name="ban" size={16} className="mt-0.5 shrink-0" />
              <div className="min-w-0"><p className="text-sm font-medium">The run cannot start yet</p><p className="text-xs opacity-80">{assign.blocked}</p></div>
            </div>
          </if>
          <if when={!assign.ready}>
            <p className="mt-3 text-xs text-muted-foreground">Still to fill in: <span className="font-mono">{list.join(value: assign.missing, separator: ", ")}</span></p>
          </if>
          <Button
            className="mt-4 shadow-lg shadow-brand/20 transition-shadow hover:shadow-brand/30"
            fullWidth={true}
            size="lg"
            icon={assign.starting ? "loader" : "play"}
            disabled={assign.starting || !assign.ready || assign.blocked != null || homing.running || releasing.running || deps.robot.trajectories.gripping || list.len(value: args.loaded_slides ?? []) == 0}
            onClick={assign.start()}
          >{assign.starting ? "Starting" : "Start the run"}</Button>
        </div>
      </div>
    </div>
  </div>
</Blok>
"""


# --------------------------------------------------------------------------- #
# The workflow                                                                 #
# --------------------------------------------------------------------------- #
@lab_app.workflow(
    logo=PROCESS_LOGO,
    bloks=[
        ActionBlok(
            TASK_PANEL,
            description="The run: the lab in 3D with the device at work glowing, and beside it the step, each device, every slide's rounds and stain, and the latest mosaic.",
        ),
        ActionBlok(
            LAB,
            description="The run in the lab: each device where it stands, glowing while it is used; the slide's route on request.",
        ),
        ActionBlok(
            PREPARE,
            description="Before a run: the lab as it stands, each device's readiness with the arm sent home or its gripper released, the arguments, what the run comes to, and the start.",
        ),
    ],
)
def process_slides(
    robot: ArkirinoLike,
    opentrons: OT2Like,
    frame: FrameLike,
    stitcher: StitcherLike,
    segmenter: SegmenterLike,
    analyzer: AnalyzerLike,
    loaded_slides: Annotated[
        list[Slide], Description("The slides to process, in order")
    ] = DEFAULT_SLIDES,
    opentrons_trajectory: Annotated[
        str,
        withStateChoices("robot.trajectories.available_trajectories"),
        Description("The trajectory that reaches the liquid handler's deck"),
    ] = "opentrons",
    microscope_trajectory: Annotated[
        str,
        withStateChoices("robot.trajectories.available_trajectories"),
        Description("The trajectory that reaches the microscope's stage"),
    ] = "microscope",
    home_trajectory: Annotated[
        str,
        withStateChoices("robot.trajectories.available_trajectories"),
        Description("Where the arm parks when all slides are done"),
    ] = "home",
    tiles: Annotated[int, Description("How many tiles to image per slide")] = 3,
    target_stain: Annotated[
        float,
        Description("The stained share of the cell pixels (0 to 1) a slide must reach"),
    ] = 0.10,
    max_rounds: Annotated[
        int, Description("How many times a slide is stained at most")
    ] = 3,
    *,
    task: Task,
    state: LabState,
) -> Generator[SlideResult, None, None]:
    """Process Slides

    Stains each slide, images it in tiles, stitches, segments and measures it, and
    stains it again while it is below the target, up to max_rounds. Then puts it back.
    Yields one result per slide.
    """
    # A new run starts with no failure on the board.
    state.failure = ""
    state.failed_slide = ""
    state.failed_step = ""
    state.failed_index = -1
    state.current_slide = ""
    state.phase = "idle"
    state.phase_index = -1
    try:
        names = [slide.name for slide in loaded_slides]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Slide names must be unique, the board is keyed by them: {names}"
            )
        if max_rounds < 1:
            raise ValueError(
                "max_rounds must be at least 1: every slide is stained once"
            )
        if len(loaded_slides) > BAY_SLOTS:
            raise ValueError(
                f"The bay holds {BAY_SLOTS} slides, {len(loaded_slides)} were loaded"
            )
        # a slide keeps its slot if it is valid and free, otherwise it gets the next free
        # one
        taken: set[int] = set()
        for slide in loaded_slides:
            if 1 <= slide.slot <= BAY_SLOTS and slide.slot not in taken:
                taken.add(slide.slot)
            else:
                slide.slot = 0
        free = iter(slot for slot in range(1, BAY_SLOTS + 1) if slot not in taken)
        for slide in loaded_slides:
            if slide.slot == 0:
                slide.slot = next(free)
        slots = [slide.slot for slide in loaded_slides]

        total = max(1, len(loaded_slides))
        state.slides = names
        state.slide_status = dict.fromkeys(names, "queued")
        state.slide_rounds = dict.fromkeys(names, 0)
        state.slide_stain = {}
        state.slides_done = 0
        state.target_stain = target_stain
        state.bay = sorted(slots)
        state.latest_stitched = None
        state.latest_labels = None
        state.latest_cells = None

        keys = [key for key, _label in PHASES]

        def enter(index: int, slide: Slide, phase: str, message: str) -> None:
            """Mark a step of the ``index``-th slide in the live state and on the task.
            A further round never moves the slide's progress back."""
            state.phase = phase
            state.phase_index = keys.index(phase)
            state.slide_progress = max(
                state.slide_progress, int(100 * PHASE_START[phase])
            )
            overall = (100 * index + state.slide_progress) / total
            task.progress(int(overall), f"{slide.name}: {message}")
            print(f"[{IDENTIFIER}] {slide.name}: {message}")

        # old forms may still send to_opentrons / to_microscope
        opentrons_trajectory, microscope_trajectory, home_trajectory = (
            name.removeprefix("to_")
            for name in (opentrons_trajectory, microscope_trajectory, home_trajectory)
        )

        def pick(at: str) -> None:
            """Go to ``at`` and take the slide there."""
            robot.run_trajectory(at)
            robot.grip()

        def place(at: str) -> None:
            """Carry the slide in to ``at``, set it down, and back out of the way."""
            robot.run_trajectory(at)
            robot.ungrip()
            robot.retreat()

        for index, slide in enumerate(loaded_slides):
            state.slide_progress = 0
            state.current_slide = slide.name
            state.current_round = 0
            state.slide_status = {**state.slide_status, slide.name: "running"}

            enter(index, slide, "pickup", f"picking up from bay slot {slide.slot}")
            pick(f"bay_{slide.slot}")
            state.bay = [slot for slot in state.bay if slot != slide.slot]
            place(opentrons_trajectory)

            while True:
                state.current_round += 1
                state.slide_rounds = {
                    **state.slide_rounds,
                    slide.name: state.current_round,
                }
                enter(
                    index,
                    slide,
                    "stain",
                    f"staining, round {state.current_round} ({slide.protocol})",
                )
                opentrons.run_protocol(slide.protocol)

                enter(index, slide, "image", "carrying to the microscope")
                pick(opentrons_trajectory)
                place(microscope_trajectory)
                acquired: list[RGBImage] = []
                for tile in range(tiles):
                    enter(index, slide, "image", f"imaging tile {tile + 1}/{tiles}")
                    frame.move_to_position(f"{slide.name}_tile_{tile}")
                    acquired.append(frame.acquire_image())

                enter(index, slide, "stitch", "stitching")
                stitched = stitcher.stitch_images(acquired)
                state.latest_stitched = stitched

                enter(index, slide, "segment", "segmenting")
                labels = segmenter.segment_cells_rgb(stitched)
                state.latest_labels = labels

                enter(index, slide, "measure", "measuring the stain")
                cells, stain = analyzer.measure_cells(stitched, labels)
                state.latest_cells = cells
                state.slide_stain = {**state.slide_stain, slide.name: stain}

                if stain >= target_stain or state.current_round >= max_rounds:
                    break
                # Below target with rounds left: back to the liquid handler for another one.
                state.slide_status = {**state.slide_status, slide.name: "restaining"}
                enter(index, slide, "stain", "back to the liquid handler")
                pick(microscope_trajectory)
                place(opentrons_trajectory)

            reached = stain >= target_stain
            enter(index, slide, "return", f"putting back in bay slot {slide.slot}")
            pick(microscope_trajectory)
            place(f"bay_{slide.slot}")
            state.bay = sorted([*state.bay, slide.slot])

            state.slide_status = {
                **state.slide_status,
                slide.name: "done" if reached else "below target",
            }
            state.slides_done += 1
            yield SlideResult(
                name=slide.name,
                stitched=stitched,
                labels=labels,
                cells=cells,
                stain=stain,
                rounds=state.current_round,
                reached_target=reached,
            )

        state.current_slide = ""
        state.phase = "park"
        state.phase_index = len(PHASES)
        robot.run_trajectory(home_trajectory)
        task.progress(100, "Done")
        state.phase = "idle"
        state.phase_index = -1
    except Exception as error:
        # For the bloks. Cancellation and GeneratorExit are not Exceptions, so they do not
        # count.
        state.failure = str(error) or type(error).__name__
        state.failed_slide = state.current_slide
        state.failed_step = dict(PHASES).get(
            state.phase, "Setting up" if state.phase_index < 0 else "Parking"
        )
        state.failed_index = (
            state.phase_index if 0 <= state.phase_index < len(PHASES) else -1
        )
        state.failed_at = datetime.now(UTC).isoformat()
        if state.current_slide:
            state.slide_status = {**state.slide_status, state.current_slide: "failed"}
        state.phase = "failed"
        raise


# --------------------------------------------------------------------------- #
# The control room: the app's default panel                                    #
# --------------------------------------------------------------------------- #
# The demo button loads slide A (dapi_stain, slot 1) and slide B (he_stain, slot 2).
CONTROL_ROOM = """
<Blok kind="agent" context="panel">
  <Column gap="1rem">
    <Card>
      <CardHeader>
        <Row justify="between" align="center" gap="1rem">
          <Column gap="0.25rem">
            <CardTitle>Helsinki staining lab</CardTitle>
            <CardDescription>Stain, image, stitch, segment and measure each slide, and stain it again until it reaches its target.</CardDescription>
          </Column>
          <Button
            icon="play"
            disabled={self.tasks.process_slides.running || demo.running}
            onClick={self.process_slides(loaded_slides: [{name: "slide A", protocol: "dapi_stain", slot: 1}, {name: "slide B", protocol: "he_stain", slot: 2}], opentrons_trajectory: "opentrons", microscope_trajectory: "microscope", home_trajectory: "home", tiles: 3, target_stain: 0.1, max_rounds: 3) into demo}
          >Run the demo slides</Button>
        </Row>
      </CardHeader>
      <CardContent className="grid gap-4">
        <Row justify="between" align="center">
          <Text weight="medium">{str.length(value: self.lab.current_slide) > 0 ? self.lab.current_slide : "No slide on the arm"}</Text>
          <Row gap="0.5rem" align="center">
            <if when={self.tasks.process_slides.duration != null}>
              <Text tone="muted" size="sm">took {fmt.duration(seconds: self.tasks.process_slides.duration)}</Text>
              <else>
                <if when={self.tasks.process_slides.started_at != null}>
                  <Text tone="muted" size="sm">started <Time value={self.tasks.process_slides.started_at} relative={true} /></Text>
                </if>
              </else>
            </if>
            <Badge variant={self.tasks.process_slides.failed ? "destructive" : self.tasks.process_slides.running ? "default" : "outline"}>{self.tasks.process_slides.status}</Badge>
          </Row>
        </Row>
        <Row gap="0.375rem" wrap={true}>
          <foreach items={[{i: 0, label: "Pick up"}, {i: 1, label: "Stain"}, {i: 2, label: "Image"}, {i: 3, label: "Stitch"}, {i: 4, label: "Segment"}, {i: 5, label: "Measure"}, {i: 6, label: "Put back"}]} let="step" key={step.i}>
            <Badge variant={self.lab.failed_index == step.i ? "destructive" : self.lab.phase_index > step.i ? "secondary" : self.lab.phase_index == step.i ? "default" : "outline"}>{step.label}</Badge>
          </foreach>
        </Row>
        <Progress value={self.lab.slides_done} max={math.max(values: [1, list.len(value: self.lab.slides)])} />
        <Text tone="muted" size="sm">{self.lab.slides_done} of {list.len(value: self.lab.slides)} slides done · target {fmt.percent(value: self.lab.target_stain, digits: 0)} stained</Text>
        <timeline />
        <if when={str.length(value: self.lab.failure) > 0}>
          <div className="flex flex-col gap-2 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-destructive">
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2">
                <Icon name="alert-triangle" size={16} className="shrink-0" />
                <p className="text-sm font-medium">Failed at {self.lab.failed_step}{str.length(value: self.lab.failed_slide) > 0 ? str.concat(values: [" · ", self.lab.failed_slide]) : ""}</p>
              </div>
              <if when={str.length(value: self.lab.failed_at) > 0}>
                <span className="shrink-0 text-xs opacity-80"><Time value={self.lab.failed_at} relative={true} /></span>
              </if>
            </div>
            <p className="whitespace-pre-wrap break-words font-mono text-xs opacity-90">{self.lab.failure}</p>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" icon="copy" onClick={ui.copy(text: self.lab.failure)}>Copy</Button>
            </div>
          </div>
        </if>
        <if when={demo.failed && str.length(value: self.lab.failure) == 0}>
          <Alert variant="destructive" icon="alert-triangle" title="The demo run did not start">{demo.error}</Alert>
        </if>
        <if when={list.len(value: self.lab.slides) == 0}>
          <Empty icon="microscope" title="No run yet" description="Start Process Slides, or run the demo slides." />
          <else>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Slide</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead align="end">Rounds</TableHead>
                  <TableHead align="end">Stained</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <foreach items={self.lab.slides} let="s">
                  <TableRow>
                    <TableCell>{s}</TableCell>
                    <TableCell>
                      <match value={obj.get(value: self.lab.slide_status, key: s, default: "queued")}>
                        <case is="done"><Badge variant="outline" icon="check" className="border-success/40 bg-success-soft text-success">done</Badge></case>
                        <case is="failed"><Badge variant="destructive" icon="x-circle">failed</Badge></case>
                        <case is="below target"><Badge variant="outline" icon="alert-triangle" className="border-warning/40 bg-warning-soft text-warning">below target</Badge></case>
                        <case is="restaining"><Badge variant="outline" icon="refresh" className="border-info/40 bg-info-soft text-info">staining again</Badge></case>
                        <case is="running"><Badge variant="outline" icon="loader" className="border-info/40 bg-info-soft text-info">in progress</Badge></case>
                        <default><Badge variant="outline">queued</Badge></default>
                      </match>
                    </TableCell>
                    <TableCell align="end">{obj.get(value: self.lab.slide_rounds, key: s, default: 0)}</TableCell>
                    <TableCell align="end">{obj.has(value: self.lab.slide_stain, key: s) ? fmt.percent(value: obj.get(value: self.lab.slide_stain, key: s), digits: 1) : "–"}</TableCell>
                  </TableRow>
                </foreach>
              </TableBody>
            </Table>
          </else>
        </if>
      </CardContent>
    </Card>
    <Grid columns={2} gap="1rem" minColumnWidth="16rem">
      <Card>
        <CardHeader>
          <CardTitle>Latest mosaic</CardTitle>
          <CardDescription>The stitched tiles of the slide measured last.</CardDescription>
        </CardHeader>
        <CardContent>
          <StructureViewer value={self.lab.latest_stitched} height="20rem" emptyText="The first mosaic shows up here." />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Its labels</CardTitle>
          <CardDescription>The cells the segmenter found in it.</CardDescription>
        </CardHeader>
        <CardContent>
          <StructureViewer value={self.lab.latest_labels} height="20rem" emptyText="The labels show up here." />
        </CardContent>
      </Card>
    </Grid>
    <Card>
      <CardHeader>
        <CardTitle>Its cells</CardTitle>
        <CardDescription>One row per cell, as the analyzer measured it: its size and shape, its mean colour, and how much of it is stained.</CardDescription>
      </CardHeader>
      <CardContent>
        <StructureViewer value={self.lab.latest_cells} height="24rem" emptyText="The cell table shows up here." />
      </CardContent>
    </Card>
  </Column>
</Blok>
"""

lab_app.blok(
    f"{IDENTIFIER}-control-room",
    CONTROL_ROOM,
    description="The lab at a glance: the slide board, the step the run is at, the latest mosaic, its labels and its cell table.",
    default=True,
)

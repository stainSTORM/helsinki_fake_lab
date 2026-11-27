"""arkirino: the robot arm of the Helsinki lab."""

from dataclasses import field
from typing import Annotated

import koil
from arkitekt import ActionBlok, App, AssignPrompt, Description, Effects, Task
from arkitekt.widgets import withStateChoices
from trajectories import WAYPOINTS

# =========================================================================== #
# arkirino: the robot arm                                                     #
# =========================================================================== #
# Named trajectories and a gripper. Nothing physical: a trajectory moves six joint angles
# and the 3D view follows.
arkirino_app = App(
    "arkirino",
    "0.1.0",
    description="Fairino FR5 robot arm: saved trajectories and a gripper.",
    effects=Effects.IRREVERSIBLE,
)

# Copy of apps/fairino-controller/models/fairino_robot.glb (assets must be inside the app's
# directory). Re-copy when fairino_model.py regenerates it.
arkirino_app.register_asset("arm", "models/fairino_robot.glb")

JOINTS = [f"joint_{index}" for index in range(1, 7)]

#: Saved trajectories, solved by testing/helsinki_layout.py for the default layout. Re-solve
#: when the layout changes.
TRAJECTORIES = list(WAYPOINTS)

#: rad/s at 100 % speed
FULL_SPEED = 2.0


def _between(a: list[float], b: list[float], t: float) -> list[float]:
    return [x + (y - x) * t for x, y in zip(a, b, strict=True)]


#: Fake teaching: each confirmed teach point sends the arm to the next of these (a pick at
#: bay slot 3 carried to the OT-2).
TEACH_POINTS: list[tuple[str, list[float]]] = [
    ("above the bay", WAYPOINTS["bay_3"][0]),
    ("down at slot 3", WAYPOINTS["bay_3"][-1]),
    ("lifted clear", WAYPOINTS["bay_3"][0]),
    (
        "halfway to the OT-2",
        _between(WAYPOINTS["bay_3"][0], WAYPOINTS["opentrons"][0], 0.5),
    ),
    ("at the OT-2", WAYPOINTS["opentrons"][-1]),
]


@arkirino_app.state(
    name="trajectories"
)  # the key a workflow's `trajectories:` demand asks for
class TrajectoryState:
    """Trajectories, arm pose and gripper.

    ``joints``, ``targets`` and ``speeds`` (rad/s) are keyed ``joint_1`` to ``joint_6``; the
    3D ``Robot`` animates from joints to targets. ``taught`` holds the trajectories taught
    since start (built-in ones: end pose only). ``teaching``, ``teach_points`` and
    ``teach_places`` describe the one being taught.
    """

    available_trajectories: list[str] = field(default_factory=list)
    current_position: str = "home"
    #: index into ``current_position``'s waypoints: the last when in, 0 when backed out
    waypoint: int = 0
    gripping: bool = False
    joints: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINTS, 0.0))
    targets: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(JOINTS, 0.0)
    )
    speeds: dict[str, float] = field(default_factory=lambda: dict.fromkeys(JOINTS, 1.0))
    taught: dict[str, list[list[float]]] = field(default_factory=dict)
    teaching: str = ""
    teach_points: int = 0
    teach_places: list[str] = field(default_factory=list)


@arkirino_app.startup
def initialize_trajectories() -> TrajectoryState:
    """Start at home with an open gripper."""
    print("[arkirino] ready")
    return TrajectoryState(available_trajectories=list(TRAJECTORIES))


@arkirino_app.action
def run_trajectory(
    state: TrajectoryState,
    name: Annotated[
        str,
        withStateChoices("self.trajectories.available_trajectories"),
        Description("The saved trajectory to run"),
    ],
    speed: Annotated[int | None, Description("Percent of full speed")] = None,
    acceleration: Annotated[
        int | None, Description("Percent of full acceleration")
    ] = None,
    *,
    task: Task,
) -> None:
    """Run Trajectory

    Moves the arm along a saved named trajectory
    """
    _drive_in(state, name, speed, acceleration, task)


def _drive_in(
    state: TrajectoryState, name: str, speed: int | None, acceleration: int | None, task: Task
) -> None:
    """Back out of where the arm is, then in along trajectory ``name``."""
    if name not in state.available_trajectories:
        raise ValueError(
            f"Unknown trajectory {name!r}. Known: {state.available_trajectories}"
        )
    print(
        f"[arkirino] {state.current_position} -> {name} (speed={speed}, acc={acceleration})"
    )
    # back out the way it came, then in along the new one
    back = _points(state, state.current_position)[: state.waypoint][::-1]
    points = back + _points(state, name)
    for index, pose in enumerate(points):
        _travel(state, pose, speed)
        task.progress(
            (index + 1) * 100 // len(points),
            f"Point {index + 1}/{len(points)} to {name}",
        )
    state.current_position = name
    state.waypoint = len(_points(state, name)) - 1


@arkirino_app.action
def retreat(
    state: TrajectoryState,
    speed: Annotated[int | None, Description("Percent of full speed")] = None,
    *,
    task: Task,
) -> None:
    """Retreat

    Backs the arm out of where it set something down, the way it came, to the clear pose
    in front of that station
    """
    _back_out(state, speed, task)


def _back_out(state: TrajectoryState, speed: int | None, task: Task) -> None:
    """Back out of the station the arm is in, the way it came."""
    back = _points(state, state.current_position)[: state.waypoint][::-1]
    for index, pose in enumerate(back):
        _travel(state, pose, speed)
        task.progress(
            (index + 1) * 100 // len(back), f"Backing out of {state.current_position}"
        )
    state.waypoint = 0


def _points(state: TrajectoryState, name: str) -> list[list[float]]:
    """Waypoints of a taught or saved trajectory, [] if unknown."""
    return state.taught.get(name) or WAYPOINTS.get(name, [])


def _travel(state: TrajectoryState, pose: list[float], speed: int | None) -> None:
    """Move to ``pose`` and wait. The furthest joint goes at ``speed`` (default 30 %), the
    rest arrive with it.
    """
    furthest = max(
        abs(target - state.joints[joint])
        for joint, target in zip(JOINTS, pose, strict=True)
    )
    duration = max(0.3, furthest / (FULL_SPEED * max(1, speed or 30) / 100))
    _start_move(state, pose, duration)
    koil.sleep(duration)
    state.joints = state.targets


def _start_move(state: TrajectoryState, pose: list[float], duration: float) -> None:
    """Publish targets and speeds so all joints arrive after ``duration`` (like
    fairino-controller's travel()).
    """
    targets = dict(zip(JOINTS, pose, strict=True))
    state.speeds = {
        joint: max(abs(targets[joint] - state.joints[joint]) / duration, 1e-3)
        for joint in JOINTS
    }
    state.targets = targets


#: Task panel for teaching: the arm in 3D, the open question, teach points and joint angles.
TEACH_PANEL = """
<Blok kind="action" context="panel" chrome="none">
  <div className="@container h-full">
    <div className="flex h-full flex-col gap-4 @4xl:flex-row">
      <div className="relative h-[28rem] min-w-0 shrink-0 overflow-hidden rounded-2xl border bg-linear-to-b from-muted/50 to-background shadow-sm @4xl:h-auto @4xl:flex-1">
        <Scene camera="iso" height="100%">
          <Robot id="arm" src={self.assets.arm} joints={self.trajectories.joints} animateTo={self.trajectories.targets} speed={self.trajectories.speeds}>
            <Glow node="link_6" on={task.ask != null} color="warning" />
            <Popout node="link_6" open={task.ask != null} title="Teach point">
              <Text size="sm">Guide me to the next point</Text>
            </Popout>
          </Robot>
        </Scene>
        <div className="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2.5 rounded-full border bg-background/90 py-1.5 pr-4 pl-3 shadow-sm">
            <span className="relative flex size-2 shrink-0">
              <span className={["absolute inline-flex size-full rounded-full opacity-70", task.running && "animate-ping", task.failed ? "bg-destructive" : task.done ? "bg-success" : "bg-brand"]} />
              <span className={["relative inline-flex size-2 rounded-full", task.failed ? "bg-destructive" : task.done ? "bg-success" : "bg-brand"]} />
            </span>
            <span className="shrink-0 text-sm font-semibold tracking-tight">Teaching {args.name}</span>
            <span className="truncate text-xs text-muted-foreground">{task.message ?? "Waiting for the first point"}</span>
          </div>
          <span className={["shrink-0 rounded-full border px-3 py-1.5 text-xs font-medium uppercase tracking-wider shadow-sm", task.failed ? "border-destructive/40 bg-destructive/10 text-destructive" : task.done ? "border-success/40 bg-success-soft text-success" : task.running ? "border-brand/40 bg-brand-soft text-brand" : "bg-background/90 text-muted-foreground"]}>{task.status}</span>
        </div>
        <div className="absolute inset-x-3 bottom-3 rounded-xl border bg-background/90 px-4 py-3 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-baseline gap-2">
              <span className="text-2xl font-semibold tabular-nums tracking-tight">{self.trajectories.teach_points}</span>
              <span className="text-xs text-muted-foreground">of 5 teach points</span>
              <span className="truncate text-xs text-muted-foreground">· at <span className="font-mono">{self.trajectories.current_position}</span></span>
            </div>
            <if when={task.can.cancel}>
              <Button variant="outline" size="sm" icon="x" onClick={task.cancel()}>Cancel</Button>
            </if>
          </div>
          <div className="mt-2.5 flex gap-1.5">
            <foreach items={[1, 2, 3, 4, 5]} let="point" key={point}>
              <div className={["h-1.5 flex-1 rounded-full transition-all duration-500", self.trajectories.teach_points >= point ? "bg-brand" : "bg-muted"]} />
            </foreach>
          </div>
        </div>
      </div>

      <div className="flex min-w-0 flex-col gap-3 @4xl:min-h-0 @4xl:w-[22rem] @4xl:shrink-0 @4xl:overflow-y-auto">
        <ask />
        <if when={task.failed}>
          <div className="flex gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-destructive">
            <Icon name="alert-triangle" size={16} className="mt-0.5 shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-medium">The teaching failed</p>
              <p className="text-xs opacity-80">{task.error}</p>
            </div>
          </div>
        </if>
        <if when={task.done}>
          <div className="flex items-center gap-3 rounded-xl border border-success/40 bg-success-soft p-3 text-success">
            <Icon name="check-circle" size={16} className="shrink-0" />
            <p className="text-xs">Saved <span className="font-medium">{args.name}</span> with {self.trajectories.teach_points} points: it is one of the arm's poses now.</p>
          </div>
        </if>

        <div className="rounded-2xl border bg-card p-4 shadow-sm">
          <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Teach points</p>
          <if when={self.trajectories.teach_points == 0}>
            <p className="text-xs text-muted-foreground">None yet: the first is recorded where the arm stands when you approve.</p>
          </if>
          <div className="flex flex-col gap-2">
            <foreach items={self.trajectories.teach_places} let="place" key={place}>
              <div className="flex items-center gap-3">
                <div className="flex size-6 shrink-0 items-center justify-center rounded-full border border-success/40 bg-success-soft text-success"><Icon name="check" size={12} /></div>
                <span className="min-w-0 truncate text-sm">{place}</span>
              </div>
            </foreach>
          </div>
        </div>

        <div className="flex flex-col gap-3 rounded-2xl border bg-card p-4 shadow-sm">
          <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Joints</p>
          <foreach items={[{key: "joint_1", label: "J1", name: "base"}, {key: "joint_2", label: "J2", name: "shoulder"}, {key: "joint_3", label: "J3", name: "elbow"}, {key: "joint_4", label: "J4", name: "wrist 1"}, {key: "joint_5", label: "J5", name: "wrist 2"}, {key: "joint_6", label: "J6", name: "wrist 3"}]} let="joint" key={joint.key}>
            <div className="flex flex-col gap-1">
              <div className="flex items-baseline justify-between text-xs">
                <span><span className="font-medium">{joint.label}</span> <span className="text-muted-foreground">{joint.name}</span></span>
                <span className="font-mono tabular-nums">{fmt.number(value: math.degrees(value: obj.get(value: self.trajectories.joints, key: joint.key, default: 0)), digits: 1)}°</span>
              </div>
              <div className="flex h-1.5 overflow-hidden rounded-full bg-muted">
                <div className="flex flex-1 justify-end">
                  <div className="h-full rounded-l-full bg-brand transition-all duration-700" width={str.concat(values: [math.clamp(value: math.div(a: math.neg(a: math.degrees(value: obj.get(value: self.trajectories.targets, key: joint.key, default: 0))), b: 1.8), min: 0, max: 100), "%"])} />
                </div>
                <div className="h-full w-px bg-foreground" />
                <div className="flex flex-1">
                  <div className="h-full rounded-r-full bg-brand transition-all duration-700" width={str.concat(values: [math.clamp(value: math.div(a: math.degrees(value: obj.get(value: self.trajectories.targets, key: joint.key, default: 0)), b: 1.8), min: 0, max: 100), "%"])} />
                </div>
              </div>
            </div>
          </foreach>
        </div>
      </div>
    </div>
  </div>
</Blok>
"""


@arkirino_app.action(
    bloks=[
        ActionBlok(
            TEACH_PANEL,
            description="The teaching: the arm in 3D where it stands, the question it holds on, the teach points so far and each joint's angle.",
        ),
    ],
)
def teach_trajectory(
    state: TrajectoryState,
    name: Annotated[str, Description("What the new trajectory is called")],
    *,
    task: Task,
) -> None:
    """Teach Trajectory

    Records a new named trajectory, one teach point at a time
    """
    name = name.strip()
    if not name:
        raise ValueError("A trajectory needs a name.")
    if name in state.available_trajectories:
        raise ValueError(f"There already is a trajectory {name!r}.")
    print(f"[arkirino] teaching {name}")
    state.teaching = name
    state.teach_points = 0
    state.teach_places = []
    points: list[list[float]] = []
    try:
        for index, (place, pose) in enumerate(TEACH_POINTS):
            moved = task.hold(
                "Move the arkirino to a new teach point",
                prompt=AssignPrompt(
                    title=f"Teach point {index + 1} of {name}",
                    description="Guide the arm to where the trajectory should pass, then "
                    "approve to record that point. Deny to stop without it.",
                ),
            )
            if not moved.approved:
                break
            # fake: the arm drives itself to the next demo point
            _start_move(state, pose, 1.0)
            koil.sleep(1.0)
            state.joints = state.targets
            state.current_position = f"{name} ({place})"
            points.append(pose)
            state.teach_points = len(points)
            state.teach_places = [*state.teach_places, place]
            task.progress(
                len(points) * 100 // len(TEACH_POINTS),
                f"Point {len(points)} of {name}: {place}",
            )
            if len(points) == len(TEACH_POINTS):
                break  # all teach points are reached
            more = task.hold(
                "Add another teach point, or finish?",
                prompt=AssignPrompt(
                    title=f"{name} has {len(points)} point{'s' if len(points) > 1 else ''}",
                    description="Approve to add another teach point, deny to finish "
                    "and save the trajectory as it is.",
                ),
            )
            if not more.approved:
                break
        if not points:
            raise RuntimeError(f"Nothing was taught: {name!r} has no points.")
        state.taught = {**state.taught, name: points}
        state.available_trajectories = [*state.available_trajectories, name]
        state.current_position = name
        state.waypoint = len(points) - 1
        print(f"[arkirino] taught {name}: {len(points)} points")
    finally:
        # kept for the task's blok until the next teach
        state.teaching = ""


@arkirino_app.action
def grip(state: TrajectoryState) -> None:
    """Grip

    Closes the gripper to hold an object
    """
    koil.sleep(0.3)
    state.gripping = True
    print(f"[arkirino] gripped at {state.current_position}")


@arkirino_app.action
def ungrip(state: TrajectoryState) -> None:
    """Ungrip

    Opens the gripper to release an object
    """
    koil.sleep(0.3)
    state.gripping = False
    print(f"[arkirino] released at {state.current_position}")


# The units the real arm (fairinogale) drives, and the protocol app demands: each move goes in to
# the station, acts on the gripper and comes back out.
Station = Annotated[
    str,
    withStateChoices("self.trajectories.available_trajectories"),
    Description("The station: a saved trajectory"),
]


@arkirino_app.action
def pick_up(
    state: TrajectoryState,
    station: Station,
    sample: Annotated[str, Description("Which slot of the station (not simulated)")] = "",
    speed: Annotated[int, Description("Percent of full speed")] = 20,
    acceleration: Annotated[int, Description("Percent of full acceleration")] = 70,
    danger_speed: Annotated[int, Description("Percent at the station (not simulated)")] = 10,
    *,
    task: Task,
) -> None:
    """Pick Up

    Drives in to the station, grips the slide and comes back out with it
    """
    _drive_in(state, station, speed, acceleration, task)
    koil.sleep(0.3)
    state.gripping = True
    _back_out(state, speed, task)


@arkirino_app.action
def release_at(
    state: TrajectoryState,
    station: Station,
    sample: Annotated[str, Description("Which slot of the station (not simulated)")] = "",
    speed: Annotated[int, Description("Percent of full speed")] = 20,
    acceleration: Annotated[int, Description("Percent of full acceleration")] = 70,
    danger_speed: Annotated[int, Description("Percent at the station (not simulated)")] = 10,
    *,
    task: Task,
) -> None:
    """Release At

    Drives in to the station, sets the slide down and comes back out
    """
    _drive_in(state, station, speed, acceleration, task)
    koil.sleep(0.3)
    state.gripping = False
    _back_out(state, speed, task)


@arkirino_app.action
def open_grip(state: TrajectoryState) -> None:
    """Open Gripper"""
    koil.sleep(0.3)
    state.gripping = False


@arkirino_app.action
def home_robot(
    state: TrajectoryState,
    speed: Annotated[int, Description("Percent of full speed")] = 30,
    acceleration: Annotated[int, Description("Percent of full acceleration")] = 30,
    *,
    task: Task,
) -> None:
    """Home Robot

    Drives to the home pose
    """
    _drive_in(state, "home", speed, acceleration, task)


#: 3D scene, what a Space draws.
ARM_SCENE = """
<Blok kind="agent" context="scene">
<Scene camera="iso" height="100%">
  <Robot id="arm" src={self.assets.arm} joints={self.trajectories.joints} animateTo={self.trajectories.targets} speed={self.trajectories.speeds}>
    <Glow node="link_6" on={self.trajectories.gripping} color="warning" />
    <Popout node="link_6" trigger="hover" title="Gripper">
      <Text size="sm">{self.trajectories.gripping ? "Holding a slide" : "Open"} at {self.trajectories.current_position}</Text>
    </Popout>
  </Robot>
</Scene>
</Blok>
"""

#: Panel: poses, speed, joint gauges, gripper.
ARM_PANEL = """
<Blok kind="agent" context="panel">
  <local name="speed" kind="STRING" default="30" />
  <div className="flex flex-col gap-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex flex-col gap-0.5">
        <p className="text-base font-semibold tracking-tight">Fairino FR5</p>
        <p className="text-sm text-muted-foreground">{self.trajectories.current_position}</p>
      </div>
      <div className="flex items-center gap-2">
        <if when={self.trajectories.gripping}>
          <Badge variant="outline" icon="box" className="border-warning/40 bg-warning-soft text-warning">holding</Badge>
        </if>
        <Badge variant={self.tasks.run_trajectory.running ? "default" : "outline"} icon={self.tasks.run_trajectory.running ? "loader" : "check"}>{self.tasks.run_trajectory.running ? "moving" : "still"}</Badge>
      </div>
    </div>

    <if when={self.tasks.run_trajectory.running}>
      <div className="flex flex-col gap-2 rounded-xl border border-brand/40 bg-brand-soft p-3">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">{self.tasks.run_trajectory.message ?? "Moving"}</span>
          <span className="font-mono text-muted-foreground">{self.tasks.run_trajectory.progress}%</span>
        </div>
        <Progress value={self.tasks.run_trajectory.progress} max={100} />
      </div>
    </if>

    <if when={!(self.trajectories.teaching == "")}>
      <Alert icon="flag" title={str.concat(values: ["Teaching ", self.trajectories.teaching])}>{self.trajectories.teach_points} of 5 teach points recorded. Answer the held task to add the next one or finish.</Alert>
    </if>

    <div className="flex flex-wrap items-start gap-4">
      <div className="flex min-w-[15rem] grow basis-[16rem] flex-col gap-3 rounded-xl border bg-card p-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-sm font-medium">Poses</p>
          <ToggleGroup bind={speed} size="sm" variant="outline">
            <ToggleGroupItem value="10" label="10%" />
            <ToggleGroupItem value="30" label="30%" />
            <ToggleGroupItem value="60" label="60%" />
            <ToggleGroupItem value="100" label="100%" />
          </ToggleGroup>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <foreach items={self.trajectories.available_trajectories} let="name" key={name}>
            <Button variant={name == self.trajectories.current_position ? "default" : "outline"} size="sm" icon={name == "home" ? "home" : "map-pin"} className="grow basis-[40%]" disabled={self.tasks.run_trajectory.running} onClick={self.run_trajectory(name: name, speed: val.to_number(value: speed) ?? 30) into moving}>{name}</Button>
          </foreach>
        </div>
        <div className="flex flex-col gap-2 border-t pt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Gripper</span>
            <span className={self.trajectories.gripping ? "text-xs font-medium text-warning" : "text-xs text-muted-foreground"}>{self.trajectories.gripping ? "closed" : "open"}</span>
          </div>
          <div className="flex gap-1.5">
            <Button variant={self.trajectories.gripping ? "default" : "outline"} icon="minimize" className="flex-1" disabled={self.trajectories.gripping || gripping.running} onClick={self.grip() into gripping}>Grip</Button>
            <Button variant={self.trajectories.gripping ? "outline" : "default"} icon="maximize" className="flex-1" disabled={!self.trajectories.gripping || releasing.running} onClick={self.ungrip() into releasing}>Release</Button>
          </div>
        </div>
      </div>

      <div className="flex min-w-[15rem] grow basis-[16rem] flex-col gap-3 rounded-xl border bg-card p-3">
        <p className="text-sm font-medium">Joints</p>
        <foreach items={[{key: "joint_1", label: "J1", name: "base"}, {key: "joint_2", label: "J2", name: "shoulder"}, {key: "joint_3", label: "J3", name: "elbow"}, {key: "joint_4", label: "J4", name: "wrist 1"}, {key: "joint_5", label: "J5", name: "wrist 2"}, {key: "joint_6", label: "J6", name: "wrist 3"}]} let="joint" key={joint.key}>
          <div className="flex flex-col gap-1">
            <div className="flex items-baseline justify-between text-xs">
              <span><span className="font-medium">{joint.label}</span> <span className="text-muted-foreground">{joint.name}</span></span>
              <span className="font-mono">{fmt.number(value: math.degrees(value: obj.get(value: self.trajectories.joints, key: joint.key, default: 0)), digits: 1)}°{self.tasks.run_trajectory.running ? str.concat(values: [" → ", fmt.number(value: math.degrees(value: obj.get(value: self.trajectories.targets, key: joint.key, default: 0)), digits: 1), "°"]) : ""}</span>
            </div>
            <div className="flex h-1.5 overflow-hidden rounded-full bg-muted">
              <div className="flex flex-1 justify-end">
                <div className="h-full rounded-l-full bg-brand" width={str.concat(values: [math.clamp(value: math.div(a: math.neg(a: math.degrees(value: obj.get(value: self.trajectories.joints, key: joint.key, default: 0))), b: 1.8), min: 0, max: 100), "%"])} />
              </div>
              <div className="h-full w-px bg-foreground" />
              <div className="flex flex-1">
                <div className="h-full rounded-r-full bg-brand" width={str.concat(values: [math.clamp(value: math.div(a: math.degrees(value: obj.get(value: self.trajectories.joints, key: joint.key, default: 0)), b: 1.8), min: 0, max: 100), "%"])} />
              </div>
            </div>
          </div>
        </foreach>
      </div>
    </div>

    <if when={moving.failed}>
      <Alert variant="destructive" icon="alert-triangle" title="Move failed">{moving.error}</Alert>
    </if>
  </div>
</Blok>
"""

arkirino_app.blok(
    "arkirino-panel",
    ARM_PANEL,
    description="Poses, joints and gripper.",
    default=True,
)

arkirino_app.blok(
    "arkirino-3d",
    ARM_SCENE,
    description="The FR5 in 3D.",
    default=True,
)


#: Logo: arm holding a slide.
ARM_LOGO = """
<Blok kind="agent" context="logo">
  <Logo viewBox="0 0 64 64" title="Arkirino">
    <Defs>
      <LinearGradient id="arm-ground" x1="0" y1="0" x2="1" y2="1">
        <Stop offset="0" stopColor="ink" />
        <Stop offset="1" stopColor="brand-strong" />
      </LinearGradient>
      <RadialGradient id="arm-glow" cx="0.5" cy="0.45" r="0.6">
        <Stop offset="0" stopColor="spectrum-orange" stopOpacity="0.45" />
        <Stop offset="1" stopColor="spectrum-orange" stopOpacity="0" />
      </RadialGradient>
      <ClipPath id="arm-tile"><Rect x="0" y="0" width="64" height="64" rx="14" /></ClipPath>
    </Defs>
    <G clipPath="url(#arm-tile)">
      <Rect x="0" y="0" width="64" height="64" fill="url(#arm-ground)" />
      <Rect x="0" y="0" width="64" height="64" fill="url(#arm-glow)" />
      <Rect x="8" y="53" width="48" height="2.5" rx="1.25" fill="surface" fillOpacity="0.25" />
      <Path d="M13 53h18l-2.5-7h-13z" fill="surface" fillOpacity="0.9" />
      <G stroke="surface" strokeLinecap="round" fill="none">
        <Line x1="22" y1="44" x2="33" y2="22" strokeWidth="6" />
        <Line x1="33" y1="22" x2="48" y2="29" strokeWidth="5" />
      </G>
      <Circle cx="22" cy="44" r="4" fill="spectrum-orange" />
      <Circle cx="33" cy="22" r="3.5" fill="spectrum-orange" />
      <Circle cx="48" cy="29" r="3" fill="spectrum-orange" />
      <G animate="float">
        <Rect x="44" y="32" width="8" height="2.5" rx="1" fill="surface" />
        <Path d="M45 34.5v5M51 34.5v5" stroke="surface" strokeWidth="2.2" strokeLinecap="round" />
        <Rect x="42.5" y="40.5" width="11" height="3" rx="0.8" fill="spectrum-blue" />
      </G>
    </G>
    <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
  </Logo>
</Blok>
"""

arkirino_app.blok(
    "arkirino-logo",
    ARM_LOGO,
    description="Logo.",
    default=True,
)

"""ot2: the liquid handler of the Helsinki lab."""

from dataclasses import field
from typing import Annotated

import koil
from arkitekt import App, Description, Effects, Task
from arkitekt.widgets import withStateChoices

# =========================================================================== #
# ot2: the Opentrons                                                          #
# =========================================================================== #
# A protocol is a list of named steps. The state says which exist and which is running.
ot2_app = App(
    "ot2",
    "0.1.0",
    description="Opentrons OT-2 running staining protocols.",
    effects=Effects.IRREVERSIBLE,
)

# OT-2 reference model, in metres. testdata/opentrons_parts.mjs regroups
# testdata/opentrons.glb so the scene can bind ``gantry_x`` > ``gantry_y`` (the head) >
# ``pipette_left``/``pipette_right``.
ot2_app.register_asset("ot2", "models/opentrons_parts.glb")

#: Head positions, one per step, cycling. (x, y) in metres from park.
DECK_SPOTS = [
    (-0.16, -0.14),
    (-0.06, 0.0),
    (0.04, 0.14),
    (-0.16, 0.14),
    (0.04, -0.14),
]

#: Deck slot under each of DECK_SPOTS (1-3 are the front row, 12 is the trash).
DECK_SLOTS = [1, 5, 9, 7, 3]

#: How far a pipette goes down to reach the deck, in metres.
PIPETTE_REACH = 0.09

PROTOCOLS = {
    "dapi_stain": [
        "Aspirate DAPI",
        "Dispense on slide",
        "Incubate",
        "Wash",
        "Wash again",
    ],
    "he_stain": ["Hematoxylin", "Rinse", "Eosin", "Dehydrate", "Clear"],
    "wash_only": ["Aspirate buffer", "Wash", "Dry"],
}


@ot2_app.state(name="run")  # the key a workflow's `run:` demand asks for
class RunState:
    """Which protocols the deck can run, and what it is running."""

    available_protocols: list[str] = field(default_factory=list)
    current_protocol: str = ""
    current_step: str = ""
    runs_completed: int = 0
    #: steps per protocol, for the panel
    protocol_steps: dict[str, list[str]] = field(default_factory=dict)
    #: running step, -1 if none
    step_index: int = -1
    #: head offset from park and left pipette depth, in metres
    head_x: float = 0.0
    head_y: float = 0.0
    pipette_down: float = 0.0
    #: The deck slot the head is over (1-12), 0 while it is parked.
    deck_slot: int = 0


@ot2_app.startup
def initialize_run() -> RunState:
    """Start with an idle deck."""
    print("[ot2] ready")
    return RunState(
        available_protocols=list(PROTOCOLS),
        protocol_steps={name: list(steps) for name, steps in PROTOCOLS.items()},
    )


@ot2_app.action
def run_protocol(
    state: RunState,
    protocol: Annotated[
        str,
        withStateChoices("self.run.available_protocols"),
        Description("The protocol to run on whatever is on the deck"),
    ],
    *,
    task: Task,
) -> None:
    """Run Protocol

    Runs a staining protocol on the slide on the deck
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}. Known: {list(PROTOCOLS)}")
    steps = PROTOCOLS[protocol]
    state.current_protocol = protocol
    print(f"[ot2] running {protocol}")
    for index, step in enumerate(steps):
        state.current_step = step
        state.step_index = index
        task.progress(int(100 * (index + 1) / len(steps)), step)
        # over a spot, down, up: 0.6 s per step
        state.head_x, state.head_y = DECK_SPOTS[index % len(DECK_SPOTS)]
        state.deck_slot = DECK_SLOTS[index % len(DECK_SLOTS)]
        koil.sleep(0.3)
        state.pipette_down = PIPETTE_REACH
        koil.sleep(0.2)
        state.pipette_down = 0.0
        koil.sleep(0.1)
    state.head_x, state.head_y = 0.0, 0.0
    state.deck_slot = 0
    state.step_index = -1
    state.current_protocol = ""
    state.current_step = ""
    state.runs_completed += 1


#: 3D scene. Model Y is up, so deck y maps to model Z.
OT2_SCENE = """
<Blok kind="agent" context="scene">
<Scene camera="iso" height="100%">
  <Model id="ot2" src={self.assets.ot2}>
    <Move node="gantry_x" axis="x" value={self.run.head_x} />
    <Move node="gantry_y" axis="z" value={self.run.head_y} />
    <Move node="pipette_left" axis="y" scale={-1} value={self.run.pipette_down} />
    <Glow on={str.length(value: self.run.current_protocol) > 0} color="brand" />
    <Popout trigger="hover" title="OT-2">
      <Text size="sm">{str.length(value: self.run.current_protocol) > 0 ? self.run.current_protocol : "Idle"}</Text>
      <Text size="sm" tone="muted">{self.run.current_step} · {self.run.runs_completed} runs</Text>
    </Popout>
  </Model>
</Scene>
</Blok>
"""

#: Panel: the deck from above, and a card per protocol with its steps and a run button.
OT2_PANEL = """
<Blok kind="agent" context="panel">
  <div className="flex flex-col gap-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex flex-col gap-0.5">
        <p className="text-base font-semibold tracking-tight">Opentrons OT-2</p>
        <p className="text-sm text-muted-foreground">{self.run.runs_completed} runs</p>
      </div>
      <Badge variant={self.tasks.run_protocol.running ? "default" : "outline"} icon={self.tasks.run_protocol.running ? "loader" : "check"}>{self.tasks.run_protocol.running ? self.run.current_protocol : "idle"}</Badge>
    </div>

    <if when={self.tasks.run_protocol.running}>
      <div className="flex flex-col gap-2 rounded-xl border border-brand/40 bg-brand-soft p-3">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">{self.run.current_step}</span>
          <span className="text-muted-foreground">step {math.add(a: self.run.step_index, b: 1)} of {list.len(value: obj.get(value: self.run.protocol_steps, key: self.run.current_protocol, default: []))}</span>
        </div>
        <Progress value={self.tasks.run_protocol.progress} max={100} />
      </div>
    </if>

    <div className="flex flex-wrap items-start gap-4">
      <div className="flex min-w-[15rem] grow basis-[16rem] flex-col gap-3 rounded-xl border bg-card p-3">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium">Deck</p>
          <span className="text-xs text-muted-foreground">{self.run.deck_slot > 0 ? str.concat(values: ["slot ", self.run.deck_slot]) : "parked"}</span>
        </div>
        <div className="flex flex-col gap-1.5">
          <foreach items={[{row: "back", slots: [{n: 10, label: ""}, {n: 11, label: ""}, {n: 12, label: "Trash"}]}, {row: "third", slots: [{n: 7, label: "H&E"}, {n: 8, label: ""}, {n: 9, label: "Buffer"}]}, {row: "second", slots: [{n: 4, label: ""}, {n: 5, label: "Tips"}, {n: 6, label: ""}]}, {row: "front", slots: [{n: 1, label: "Slide"}, {n: 2, label: ""}, {n: 3, label: "DAPI"}]}]} let="row" key={row.row}>
            <div className="flex gap-1.5">
              <foreach items={row.slots} let="slot" key={slot.n}>
                <div className={self.run.deck_slot == slot.n ? (self.run.pipette_down > 0 ? "flex h-14 flex-1 flex-col justify-between rounded-lg border-2 border-brand bg-brand-soft p-1.5 animate-pulse" : "flex h-14 flex-1 flex-col justify-between rounded-lg border-2 border-brand bg-brand-soft p-1.5") : (str.length(value: slot.label) > 0 ? "flex h-14 flex-1 flex-col justify-between rounded-lg border bg-muted p-1.5" : "flex h-14 flex-1 flex-col justify-between rounded-lg border border-dashed p-1.5")}>
                  <span className="text-[10px] font-medium text-muted-foreground">{slot.n}</span>
                  <span className="truncate text-xs font-medium">{slot.label}</span>
                </div>
              </foreach>
            </div>
          </foreach>
        </div>
      </div>

      <div className="flex min-w-[15rem] grow basis-[18rem] flex-col gap-2">
        <foreach items={self.run.available_protocols} let="protocol" key={protocol}>
          <div className={protocol == self.run.current_protocol ? "flex flex-col gap-2 rounded-xl border border-brand bg-card p-3 shadow-sm" : "flex flex-col gap-2 rounded-xl border bg-card p-3"}>
            <div className="flex items-center justify-between gap-2">
              <div className="flex flex-col">
                <span className="text-sm font-medium">{protocol}</span>
                <span className="text-xs text-muted-foreground">{list.len(value: obj.get(value: self.run.protocol_steps, key: protocol, default: []))} steps</span>
              </div>
              <Button size="sm" variant={protocol == self.run.current_protocol ? "default" : "outline"} icon="play" disabled={self.tasks.run_protocol.running} onClick={self.run_protocol(protocol: protocol) into staining}>Run</Button>
            </div>
            <div className="flex flex-wrap gap-1">
              <foreach items={list.range(start: 0, end: list.len(value: obj.get(value: self.run.protocol_steps, key: protocol, default: [])))} let="i" key={i}>
                <span className={protocol != self.run.current_protocol ? "rounded-md bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground" : (i < self.run.step_index ? "rounded-md bg-success-soft px-1.5 py-0.5 text-[11px] text-success" : (i == self.run.step_index ? "rounded-md bg-brand px-1.5 py-0.5 text-[11px] text-primary-foreground" : "rounded-md bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground"))}>{list.at(value: obj.get(value: self.run.protocol_steps, key: protocol, default: []), index: i)}</span>
              </foreach>
            </div>
          </div>
        </foreach>
      </div>
    </div>

    <if when={staining.failed}>
      <Alert variant="destructive" icon="alert-triangle" title="Protocol failed">{staining.error}</Alert>
    </if>
  </div>
</Blok>
"""

ot2_app.blok(
    "ot2-panel",
    OT2_PANEL,
    description="Deck and protocols.",
    default=True,
)

ot2_app.blok(
    "ot2-3d",
    OT2_SCENE,
    description="The OT-2 in 3D.",
    default=True,
)


#: Logo: pipette over a plate.
OT2_LOGO = """
<Blok kind="agent" context="logo">
  <Logo viewBox="0 0 64 64" title="OT-2">
    <Defs>
      <LinearGradient id="ot2-ground" x1="0" y1="0" x2="1" y2="1">
        <Stop offset="0" stopColor="ink" />
        <Stop offset="1" stopColor="brand-strong" />
      </LinearGradient>
      <RadialGradient id="ot2-glow" cx="0.5" cy="0.45" r="0.6">
        <Stop offset="0" stopColor="spectrum-blue" stopOpacity="0.45" />
        <Stop offset="1" stopColor="spectrum-blue" stopOpacity="0" />
      </RadialGradient>
      <ClipPath id="ot2-tile"><Rect x="0" y="0" width="64" height="64" rx="14" /></ClipPath>
    </Defs>
    <G clipPath="url(#ot2-tile)">
      <Rect x="0" y="0" width="64" height="64" fill="url(#ot2-ground)" />
      <Rect x="0" y="0" width="64" height="64" fill="url(#ot2-glow)" />
      <Rect x="9" y="11" width="46" height="43" rx="4" fill="none" stroke="surface" strokeOpacity="0.45" strokeWidth="1.5" />
      <Rect x="9" y="16" width="46" height="4" rx="1" fill="surface" fillOpacity="0.85" />
      <Rect x="26" y="18" width="12" height="9" rx="2" fill="surface" />
      <Path d="M30.4 27h3.2l-.7 7.5h-1.8z" fill="surface" />
      <Path d="M32 37.2c-1.3 1.7-1.9 2.8-1.9 3.6a1.9 1.9 0 0 0 3.8 0c0-.8-.6-1.9-1.9-3.6z" fill="spectrum-blue" animate="float" />
      <Rect x="14" y="44.5" width="36" height="6.5" rx="1.5" fill="surface" fillOpacity="0.2" />
      <G fill="spectrum-blue">
        <Circle cx="19" cy="47.75" r="1.7" fillOpacity="0.9" />
        <Circle cx="25.5" cy="47.75" r="1.7" fillOpacity="0.6" />
        <Circle cx="32" cy="47.75" r="1.7" fillOpacity="0.9" />
        <Circle cx="38.5" cy="47.75" r="1.7" fillOpacity="0.35" />
        <Circle cx="45" cy="47.75" r="1.7" fillOpacity="0.6" />
      </G>
    </G>
    <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
  </Logo>
</Blok>
"""

ot2_app.blok(
    "ot2-logo",
    OT2_LOGO,
    description="Logo.",
    default=True,
)

"""lab.py: serve the Helsinki lab, all seven apps in one process.

The six devices and tools (``arkirino``, ``ot2``, ``openuc``, ``stitcher``, ``starmist``,
``analyzer``) and the workflow app ``helsinki-protocolapp`` each live in their own folder,
an arkitekt project of its own. This loads every ``<folder>/app.py`` and serves them
together.

The segmenter is ``zellpose`` (~/Code/apps/zellpose, needs cellpose and torch) and runs
separately: ``cd ~/Code/apps/zellpose && uv run arkitekt run dev``. Without it
``process_slides`` does not bind and ``--demo`` does not start.

Run:  uv run lab.py            serve all seven (Ctrl-C stops them)
      uv run lab.py --demo     ... and run process_slides once
      uv run lab.py --check    offline: do the demands match the apps?
      uv run --with requests lab.py --check --real
                               ... with the real arm and OT-2 from the repos next to this one
      uv run lab.py --force    take over agents a killed run left behind

Each app logs in on its own, so a first run shows seven device-code prompts at once.
"""

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from arkitekt import App, Runtime, check_apps, providing
from arkitekt_runtime.types import TaskPolicy
from rekuest.client.client import RekuestCaller

HERE = Path(__file__).resolve().parent

#: zellpose's checkout, next to this repo.
ZELLPOSE = HERE.parent / "zellpose"

#: The real arm and liquid handler, next to this repo: ``--check --real`` checks them in place of
#: the simulated ones, under the names the protocol app demands (on the Pi: ``app:`` in their
#: config.yaml).
REAL = {"arkirino": HERE.parent / "fairinogale", "ot2": HERE.parent / "OT2windy_service"}


def load(folder: Path) -> ModuleType:
    """``<folder>/app.py`` as the module ``<folder name>_app``, with its folder importable."""
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(f"{folder.name}_app", folder / "app.py")
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"No app at {folder / 'app.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def the_app(module: ModuleType) -> App[Any]:
    """The module's only App."""
    (app,) = (v for v in vars(module).values() if isinstance(v, App))
    return app


#: The apps the protocol app composes, by folder.
PROVIDERS: list[App[Any]] = [
    the_app(load(HERE / name))
    for name in ("arkirino", "ot2", "openuc", "stitcher", "starmist", "analyzer")
]
protocolapp = load(HERE / "protocolapp")
lab_app: App[Any] = protocolapp.lab_app

#: What --demo loads: (name, protocol, bay slot) per slide.
DEMO_SLIDES = [("slide A", "dapi_stain", 1), ("slide B", "he_stain", 2)]

#: How long one app may take to start and register, once it is logged in.
START_TIMEOUT = 30


async def run_demo(runtime: Runtime[Any]) -> None:
    """Call ``process_slides`` once and print each result as it is yielded.

    ``policy=AUTO``: under the default (ASK) the server would hold every device call for
    approval, and nobody is watching.
    """
    rekuest = runtime.require(RekuestCaller)
    slides = [protocolapp.Slide(name=n, protocol=p, slot=b) for n, p, b in DEMO_SLIDES]
    print("\n--- process_slides ---------------------------------------", flush=True)
    try:
        async for result in rekuest.aiterate(
            protocolapp.process_slides, loaded_slides=slides, policy=TaskPolicy.AUTO
        ):
            print(f"  {result}", flush=True)
    except Exception as error:  # noqa: BLE001 -- a failed run leaves the lab up to inspect it
        print(f"\n[lab] the demo run failed: {error!r}", flush=True)
        return
    print("\n[lab] the demo run is done; every app is still serving -- Ctrl-C to stop", flush=True)


async def main(*, demo: bool, force: bool) -> None:
    """Serve all seven apps; with ``demo``, run the workflow once they are all registered."""
    async with providing(
        *PROVIDERS,
        lab_app,
        headless=True,
        force=force,
        timeout=START_TIMEOUT,
        check=True,
    ) as running:
        print("[lab] all seven apps are registered", flush=True)
        refused = await running.check()
        for unmet in refused:
            print(f"[lab] would not bind: {unmet}", flush=True)
        if demo and not refused:
            await run_demo(running[lab_app])
        await running.wait()


def check(real: bool = False) -> int:
    """Compare every demand of the protocol app with the app meant to serve it, offline.

    zellpose's App is read from its checkout; declaring it does not import cellpose.
    """
    providers = list(PROVIDERS)
    if ZELLPOSE.exists():  # without the checkout its demand is noted, not compared
        providers.append(the_app(load(ZELLPOSE)))
    if real:
        providers = [p for p in providers if p.identifier not in REAL]
        for name, folder in REAL.items():
            device = the_app(load(folder))
            device.identifier = name
            providers.append(device)
    findings = [f for f in check_apps(*providers, lab_app) if f.level == "error"]
    for finding in findings:
        print(finding)
    print("ALL DEMANDS MATCH" if not findings else f"{len(findings)} DEMAND(S) UNMET")
    return 1 if findings else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="run process_slides once when everything serves")
    parser.add_argument("--force", action="store_true", help="take over agents a killed run left registered")
    parser.add_argument("--check", action="store_true", help="only compare the demands with the apps, offline")
    parser.add_argument("--real", action="store_true", help="with --check: the real arm and OT-2 next to this repo")
    options = parser.parse_args()
    if options.check:
        sys.exit(check(options.real))
    try:
        asyncio.run(main(demo=options.demo, force=options.force))
    except KeyboardInterrupt:
        print("\n[lab] stopped", flush=True)

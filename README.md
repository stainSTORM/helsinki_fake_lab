# helsinki-lab

The Helsinki lab workflow and the apps it drives, one arkitekt project per folder.

| folder | app | |
|---|---|---|
| `protocolapp` | `helsinki-protocolapp` | the workflow, `process_slides` |
| `arkirino` | `arkirino` | robot arm |
| `ot2` | `ot2` | liquid handler |
| `openuc` | `openuc` | microscope |
| `stitcher` | `stitcher` | tiles into a mosaic |
| `starmist` | `starmist` | segmenter (served, not used by the workflow) |
| `analyzer` | `analyzer` | cell table and stained share |

The segmenter the workflow uses is `zellpose` (`~/Code/apps/zellpose`), run on its own.

Run one app: `cd <folder> && uv run arkitekt run dev`

Run all seven in one process (root project):

```
uv run lab.py            # serve
uv run lab.py --demo     # serve and run process_slides once
uv run lab.py --check    # offline: do the workflow's demands match the apps?
```

The arm's trajectories and the sample bay model are written by `~/Code/testing/helsinki_layout.py`.

"""Embodiment presets — every number an upstream config or the EmbodiedScore
protocol fixes, nothing derived.

STANDARD  EmbodiedScore's shared body: the AgentCanvas workspace's bare VLN-CE R2R
          rig (``coding-agent/configs/habitat/std_task.yaml``: 0.25 m / 15°, 1.5 m /
          0.1 m agent, sliding on, RGB 512² + depth 256² at hfov 90 and 1.25 m,
          scene navmesh file) plus a 30° tilt clamped to ±60° so every line has
          LOOK_UP / LOOK_DOWN. Decided 2026-09-07.
VLNCE     VLN-CE ``habitat_extensions/config/vlnce_task.yaml`` (R2R-CE; IVLN-CE
          reuses it): as STANDARD but RGB 224² and no tilt.
RXR_CE    VLN-CE ``rxr_vlnce_english_task.yaml``: the LoCoBot rig on habitat-lab
          0.1.7 — 0.25 m / 30°, tilt 30° unlimited, 0.88 m / 0.18 m, sliding off,
          RGB-D 640x480 hfov 79 at 0.88 m, scene navmesh file (0.1.7 never
          recomputes).
LOCOBOT   habitat-lab ``benchmark/nav/objectnav/objectnav_hm3d.yaml`` (== mp3d):
          RXR_CE's rig with the navmesh recomputed for the 0.18 / 0.88 agent
          (habitat-lab 0.2.4 behaviour).
STRETCH   goat-bench ``goat_stretch_hm3d.yaml`` == OVON ``objectnav_stretch_hm3d.yaml``:
          0.25 m / 30°, tilt 30° unlimited, 1.41 m / 0.17 m, sliding off, RGB
          360x640 (portrait) hfov 42 at 1.31 m, no depth sensor, navmesh
          recomputed with climb 0.1 / cell 0.05 (every released experiment config
          of both repos).
EXPLORE_EQA  explore-eqa ``cfg/vlm_exp.yaml`` (HM-EQA; MemoryEQA's MT-HM3D reuses it):
          RGB-D 640x480 hfov 120 at 1.5 m, camera pitched -30° at start, the
          whole agent pitches (``pitch_moves_camera``), free-pose teleport
          locomotion; tilt 30° clamped to ±60°.
EXPRESS   EXPRESS-Bench ``fine_eqa.yaml``: RGB-D 512² hfov 90 at 1.5 m, camera
          level, free-pose teleport locomotion, no tilt.
"""

from __future__ import annotations

from ..env import Body, CameraSpec, NavMesh

STANDARD = Body(forward_step_m=0.25, turn_deg=15.0, tilt_deg=30.0, tilt_limit_deg=60.0,
                agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
                rgb=CameraSpec(512, 512, 90.0, (0.0, 1.25, 0.0)), depth=CameraSpec(256, 256, 90.0, (0.0, 1.25, 0.0)),
                navmesh=NavMesh.file())

VLNCE = Body(forward_step_m=0.25, turn_deg=15.0, agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
             rgb=CameraSpec(224, 224, 90.0, (0.0, 1.25, 0.0)), depth=CameraSpec(256, 256, 90.0, (0.0, 1.25, 0.0)),
             navmesh=NavMesh.file())

RXR_CE = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=None,
              agent_height_m=0.88, agent_radius_m=0.18, allow_sliding=False,
              rgb=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)), depth=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)),
              navmesh=NavMesh.file())

LOCOBOT = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=None,
               agent_height_m=0.88, agent_radius_m=0.18, allow_sliding=False,
               rgb=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)), depth=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)),
               navmesh=NavMesh.recompute(agent_radius=0.18, agent_height=0.88))

STRETCH = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=None,
               agent_height_m=1.41, agent_radius_m=0.17, allow_sliding=False,
               rgb=CameraSpec(360, 640, 42.0, (0.0, 1.31, 0.0)), depth=None,
               navmesh=NavMesh.recompute(agent_radius=0.17, agent_height=1.41, agent_max_climb=0.1, cell_height=0.05))

EXPLORE_EQA = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=60.0, camera_pitch_deg=-30.0,
                   pitch_moves_camera=True, locomotion="teleport",
                   agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
                   rgb=CameraSpec(640, 480, 120.0, (0.0, 1.5, 0.0)), depth=CameraSpec(640, 480, 120.0, (0.0, 1.5, 0.0)),
                   navmesh=NavMesh.file())

EXPRESS = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=None, camera_pitch_deg=0.0, locomotion="teleport",
               agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
               rgb=CameraSpec(512, 512, 90.0, (0.0, 1.5, 0.0)), depth=CameraSpec(512, 512, 90.0, (0.0, 1.5, 0.0)),
               navmesh=NavMesh.file())

__all__ = ["STANDARD", "VLNCE", "RXR_CE", "LOCOBOT", "STRETCH", "EXPLORE_EQA", "EXPRESS"]

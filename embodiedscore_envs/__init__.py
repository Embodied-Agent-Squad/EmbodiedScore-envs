"""EmbodiedScore environments — Gymnasium ids registered on import:

    EmbodiedScore/VLNCE-R2R-v0    R2R-CE (Room-to-Room, continuous environments)
    EmbodiedScore/VLNCE-RxR-v0    RxR-CE (guide role, en-US + en-IN by default)

Both wrap the same VLNCEEnv body; ``make_vlnce()`` assembles the board stack.
"""

from gymnasium.envs.registration import register

__version__ = "0.1.0"

register(id="EmbodiedScore/VLNCE-R2R-v0", entry_point="embodiedscore_envs.vlnce.env:VLNCEEnv",
         max_episode_steps=500, kwargs={"dataset": "r2r"})
register(id="EmbodiedScore/VLNCE-RxR-v0", entry_point="embodiedscore_envs.vlnce.env:VLNCEEnv",
         max_episode_steps=500, kwargs={"dataset": "rxr"})

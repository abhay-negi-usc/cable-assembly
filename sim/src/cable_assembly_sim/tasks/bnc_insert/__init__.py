"""BNC insertion with the plug already in the gripper (bnc_insert_env_cfg.py)."""

import gymnasium as gym

from . import agents

gym.register(
    id="CableAssembly-BNC-Insert-iiwa14-v0",
    entry_point=f"{__name__}.bnc_insert_env:BncInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bnc_insert_env_cfg:BncInsertEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BncInsertPPORunnerCfg",
    },
)

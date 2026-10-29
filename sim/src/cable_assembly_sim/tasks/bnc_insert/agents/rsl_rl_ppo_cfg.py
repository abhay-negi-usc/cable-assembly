"""PPO for the BNC insertion: Isaac Lab's NIST Factory runner (isaaclab_tasks/contrib/nist), with the
critic on the env's privileged state."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BncInsertPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 100
    experiment_name = "bnc_insert_iiwa14"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    critic = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=6e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.90,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

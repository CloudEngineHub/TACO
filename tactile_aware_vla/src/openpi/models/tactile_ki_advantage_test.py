import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import config as _config


def _dummy_config(**kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        use_force_condition=True,
        force_dim=12,
        force_history_len=2,
        force_hidden_dim=16,
        use_knowledge_insulation=True,
        use_advantage_conditioning=True,
        adv_hidden_dim=16,
        **kwargs,
    )


def _force_obs(config: pi0_config.Pi0Config, batch_size: int = 2) -> _model.Observation:
    obs = config.fake_obs(batch_size)
    if obs.force_history is None:
        obs = dataclasses.replace(obs, force_history=jnp.ones((batch_size, config.force_history_len, config.force_dim)))
    return obs


def _path_str(path) -> str:
    return "/".join(str(part) for part in path)


def test_advantage_missing_field_defaults_to_one_and_loss_runs():
    key = jax.random.key(1)
    config = _dummy_config()
    model = config.create(key)
    obs = _force_obs(config)
    act = config.fake_act(2)

    advantage = _model.get_advantage_from_batch(obs)
    assert advantage.shape == (2, 1)
    assert jnp.all(advantage == 1.0)
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act, train=True)
    assert loss.shape == (2, config.action_horizon)


def test_advantage_dropout_one_uses_null_condition():
    key = jax.random.key(2)
    config = _dummy_config(adv_dropout_prob=1.0)
    model = config.create(key)
    obs = dataclasses.replace(_force_obs(config), advantage=jnp.zeros((2, 1)))
    timestep = jnp.ones((2,))
    noisy_actions = config.fake_act(2)

    _, _, _, dropped_cond = model.embed_suffix(
        obs,
        noisy_actions,
        timestep,
        train=True,
        adv_dropout_rng=key,
    )
    _, _, _, uncond = model.embed_suffix(
        obs,
        noisy_actions,
        timestep,
        force_unconditional_advantage=True,
    )
    assert jnp.allclose(dropped_cond, uncond)


def test_ki_trainable_filter_freezes_vlm_and_keeps_action_side_trainable():
    model_config = _dummy_config()
    train_config = _config.TrainConfig(
        name="ki_filter_test",
        model=model_config,
        exp_name="debug",
        wandb_enabled=False,
    )
    abstract_model = nnx.eval_shape(model_config.create, jax.random.key(3))
    params = nnx.state(abstract_model, nnx.Param)
    trainable = {_path_str(path) for path in params.filter(train_config.trainable_filter).flat_state()}
    frozen = {
        _path_str(path)
        for path in params.filter(nnx.All(nnx.Param, train_config.effective_freeze_filter)).flat_state()
    }

    assert any(path.startswith("PaliGemma/img/") for path in frozen)
    assert any(path.startswith("PaliGemma/llm/") and "_1" not in path for path in frozen)
    assert not any(path.startswith("PaliGemma/img/") for path in trainable)
    assert not any(path.startswith("PaliGemma/llm/") and "_1" not in path for path in trainable)
    assert any("PaliGemma/llm/" in path and "_1" in path for path in trainable)
    assert any(path.startswith("force_encoder/") for path in trainable)
    assert any(path.startswith("advantage_encoder/") for path in trainable)
    assert any(path.startswith("null_advantage") for path in trainable)

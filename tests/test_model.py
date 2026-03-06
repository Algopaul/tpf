"""Behavioural tests for tpflow.model helpers.

These tests are intentionally model-size-agnostic: they use a tiny CFMDec /
RegressionDec so the suite stays fast.  JAX compilation happens on first call;
subsequent calls in the same process reuse the cache.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flanch.config import MLPConfig
from flanch.model import EmbMLP
from flax import nnx

from tpflow.model import (
    CFMDec,
    RegressionDec,
    _save_checkpoint,
    flow_inference,
    load_model,
    load_regression_model,
    make_flow_fn,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Tiny model dims: state_dim=4, so CFMDec input = 4+1+1 = 6, output = 4.
STATE_DIM = 4
BATCH = 8
N_COND = 5
N_ODE_STEPS = 2  # keep tests fast


@pytest.fixture(scope="module")
def cfm_model():
    cfg = MLPConfig(
        features_in=STATE_DIM + 2, features_out=STATE_DIM, features_inner=8, layers=2
    )
    model = CFMDec(EmbMLP.from_config(cfg, rngs=nnx.Rngs(0)))
    model.eval()
    return model


@pytest.fixture(scope="module")
def regression_model():
    cfg = MLPConfig(
        features_in=STATE_DIM + 2, features_out=STATE_DIM, features_inner=8, layers=2
    )
    model = RegressionDec(
        EmbMLP.from_config(cfg, rngs=nnx.Rngs(0)), time_conditioned=True
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def source_batch():
    return jnp.ones((BATCH, STATE_DIM))


@pytest.fixture(scope="module")
def cslist():
    return jnp.linspace(0.0, 1.0, N_COND)


# ---------------------------------------------------------------------------
# make_flow_fn and flow_inference produce the same outputs
# ---------------------------------------------------------------------------


def test_make_flow_fn_matches_flow_inference(cfm_model, source_batch, cslist):
    """make_flow_fn (lax.map) and flow_inference (Python loop) must agree."""
    ref = flow_inference(cfm_model, source_batch, cslist, n_steps=N_ODE_STEPS)

    run_fn = make_flow_fn(cfm_model, n_steps=N_ODE_STEPS)
    out = np.array(run_fn(source_batch, cslist))

    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_make_flow_fn_output_shape(cfm_model, source_batch, cslist):
    run_fn = make_flow_fn(cfm_model, n_steps=N_ODE_STEPS)
    out = run_fn(source_batch, cslist)
    assert out.shape == (N_COND, BATCH, STATE_DIM)


# ---------------------------------------------------------------------------
# _save_checkpoint / load roundtrip
# ---------------------------------------------------------------------------


def _tiny_cfm_cfg():
    """Return a minimal CFMTraining-like object (namespace, no Hydra needed)."""
    from types import SimpleNamespace

    mlp_cfg = MLPConfig(
        features_in=STATE_DIM + 2, features_out=STATE_DIM, features_inner=8, layers=2
    )
    return SimpleNamespace(
        model_type="mlp", mlp=mlp_cfg, data=SimpleNamespace(name="test")
    )


def _tiny_regression_cfg():
    from types import SimpleNamespace

    mlp_cfg = MLPConfig(
        features_in=STATE_DIM + 2, features_out=STATE_DIM, features_inner=8, layers=2
    )
    return SimpleNamespace(
        model_type="mlp",
        mlp=mlp_cfg,
        time_conditioned=True,
        mode="step",
    )


def test_save_and_load_cfm_model_roundtrip(tmp_path, cfm_model):
    cfg = _tiny_cfm_cfg()
    info = {
        "model_type": cfg.model_type,
        "data_name": cfg.data.name,
        "sample_shape": [STATE_DIM],
        "epoch": 1,
    }
    _save_checkpoint(
        cfm_model,
        cfg,
        epoch=1,
        sample_shape=(STATE_DIM,),
        info=info,
        output_dir=tmp_path,
    )

    loaded = load_model(tmp_path / "1")

    x = jnp.ones((4, STATE_DIM))
    t = jnp.ones((4, 1))
    p = jnp.ones((4, 1))
    np.testing.assert_allclose(
        np.array(cfm_model(x, t, p)),
        np.array(loaded(x, t, p)),
        rtol=1e-5,
    )


def test_save_and_load_regression_model_roundtrip(tmp_path, regression_model):
    cfg = _tiny_regression_cfg()
    info = {
        "model_type": cfg.model_type,
        "time_conditioned": cfg.time_conditioned,
        "mode": cfg.mode,
        "sample_shape": [STATE_DIM],
        "epoch": 1,
    }
    _save_checkpoint(
        regression_model,
        cfg,
        epoch=1,
        sample_shape=(STATE_DIM,),
        info=info,
        output_dir=tmp_path,
    )

    loaded = load_regression_model(tmp_path / "1")

    x = jnp.ones((4, STATE_DIM))
    t = jnp.ones((4, 1))
    p = jnp.ones((4, 1))
    np.testing.assert_allclose(
        np.array(regression_model(x, t, p)),
        np.array(loaded(x, t, p)),
        rtol=1e-5,
    )


def test_save_checkpoint_writes_expected_files(tmp_path, cfm_model):
    cfg = _tiny_cfm_cfg()
    info = {
        "model_type": "mlp",
        "epoch": 1,
        "sample_shape": [STATE_DIM],
        "data_name": "test",
    }
    _save_checkpoint(
        cfm_model,
        cfg,
        epoch=1,
        sample_shape=(STATE_DIM,),
        info=info,
        output_dir=tmp_path,
    )

    checkpoint_dir = tmp_path / "1"
    assert (checkpoint_dir / "config.yaml").exists()
    assert (checkpoint_dir / "checkpoint_info.json").exists()
    assert (checkpoint_dir / "state").exists()

    with open(checkpoint_dir / "checkpoint_info.json") as f:
        saved_info = json.load(f)
    assert saved_info == info

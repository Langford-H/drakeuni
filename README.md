# DrakeUni

Experimental Drake batch simulation runtime for UniLab's Drake backend adapter.

Current scope:

- Compiled C++/pybind batch pool driven by parsed MJCF model contracts.
- `nbatch` and `nthread` worker control.
- Python API modeled after MuJoCoUni's `BatchEnvPool` direction.
- Intended to be consumed by UniLab's Drake backend adapter.

This is not a full Drake backend yet. The current runtime expects a model file,
environment count, simulation step size, and worker count. Task semantics such
as base-body meaning, named sensor views, rewards, and observations belong to
the consuming UniLab backend/task layer.

## Directory Roles

`src/drakeuni/runtime/` is the Python-facing runtime interface. It owns the
public config/data contracts, MJCF contract parsing, Drake-compatible MJCF
materialization, runtime construction, and the reset/step/sensor API used by
UniLab.

`src/drakeuni/compiled/` is the native extension layer. It contains the C++
Drake batch pool source and the built pybind extension that performs batched
physics stepping.

The intended call path is:

```text
UniLab DrakeBackend
  -> drakeuni.runtime
      -> drakeuni.compiled.DrakeEnvPool
          -> Drake C++ simulation
```

## Build Batch Extension

```bash
uv run python scripts/build_drake_batch.py --drake-home /Users/huanghaochen/solver/drake/install
```

The extension is written to `src/drakeuni/compiled/_drake_env_pool*.so`.

## Install Editable

From a consuming project:

```bash
uv pip install -e /Users/huanghaochen/solver/drakeuni
```

## Runtime API

```python
from drakeuni.runtime import DrakeBatchConfig, create_runtime

runtime = create_runtime(
    DrakeBatchConfig(
        model_file="/path/to/scene_flat_drake.xml",
        num_envs=32,
        sim_dt=0.002,
        nthread=8,
    )
)
```

The preferred integration point is `drakeuni.runtime`. `DrakeEnvPool` and
the compiled extension are lower-level implementation details.

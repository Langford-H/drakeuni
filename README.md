# DrakeUni

Experimental Drake batch simulation runtime for UniLab's Drake backend adapter.

Current scope:

- Compiled C++/pybind batch pool driven by parsed MJCF model contracts.
- `nbatch` and `nthread` worker control.
- Python API modeled after MuJoCoUni's `BatchEnvPool` direction.
- Intended to be consumed by UniLab's Drake backend adapter.

This is not a full Drake backend yet. The current runtime expects UniLab tasks
to provide the model file, base body name, robot profile label, PD gains, and
worker count.

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
from drakeuni.runtime import DrakeRuntimeConfig, create_runtime

runtime = create_runtime(
    DrakeRuntimeConfig(
        model_file="/path/to/scene_flat_drake.xml",
        num_envs=32,
        sim_dt=0.002,
        mode="batch",
        base_name="base",
        robot_profile="my_robot",
        nthread=8,
    )
)
```

The preferred integration point is `drakeuni.runtime`. `DrakeEnvPool` and
the compiled extension are lower-level implementation details.

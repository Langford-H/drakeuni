# DrakeUni

Experimental Drake batch simulation runtime extracted from the UniLab Go1 Drake backend.

Current scope:

- Go1-only compiled C++/pybind batch pool.
- `nbatch` and `nthread` worker control.
- Python API modeled after MuJoCoUni's `BatchEnvPool` direction.
- Intended to be consumed by UniLab's Drake backend adapter.

This is not a generic Drake backend yet. The next design step is to remove the
remaining Go1 metadata arguments from the low-level pool constructor.

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
        nthread=8,
    )
)
```

The preferred integration point is `drakeuni.runtime`. `DrakeEnvPool` and
UniLab is being cut over to the runtime protocol.

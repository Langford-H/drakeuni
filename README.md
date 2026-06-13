# DrakeUni

Experimental Drake batch simulation runtime extracted from the UniLab Go1 Drake backend.

Current scope:

- Go1-only native C++/pybind pool.
- `nbatch` and `nthread` worker control.
- Python API modeled after MuJoCoUni's `BatchEnvPool` direction.
- Intended to be consumed by UniLab's Drake backend adapter.

This is not a generic Drake backend yet. The next design step is to remove the
remaining Go1 metadata arguments from the low-level pool constructor.

## Build Native Extension

```bash
uv run python scripts/build_drake_native.py --drake-home /Users/huanghaochen/solver/drake/install
```

The extension is written to `src/drakeuni/native/_drake_env_pool*.so`.

## Install Editable

From a consuming project:

```bash
uv pip install -e /Users/huanghaochen/solver/drakeuni
```

## API

```python
from drakeuni.batch_env import DrakeEnvPool
```

`NativeDrakeEnvPool` remains as an alias while UniLab's native Drake adapter is
being cut over.

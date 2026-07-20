"""Thin wrapper around slime's convert_hf_to_torch_dist.py for Modal volumes.

The upstream script's shutil.move(iter_0000001 -> release) only sees local shards,
which poisons the volume state for multi-node conversions (the rename propagates a
deletion of iter_0000001/ that wipes other nodes' committed shards).

Setting SKIP_RELEASE_RENAME=1 suppresses the rename so all nodes commit
to iter_0000001/ additively. Megatron loads from iter_0000001/ via the
tracker file just fine.

When SKIP_RELEASE_RENAME is unset this wrapper is a transparent pass-through.
"""

import os
import sys
from pathlib import Path

_UPSTREAM = "/root/slime/tools/convert_hf_to_torch_dist.py"

_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

with open(_UPSTREAM) as f:
    _src = f.read()

_src = _src.replace(
    "import slime_plugins.mbridge  # noqa: F401\n",
    "import slime_plugins.mbridge  # noqa: F401\n"
    "from modal_helpers.runtime_patches import apply_modal_runtime_patches\n"
    "apply_modal_runtime_patches()\n",
)

if os.environ.get("CONVERSION_DIST_TIMEOUT_MINUTES"):
    _src = _src.replace(
        "import torch.distributed as dist\n",
        "import torch.distributed as dist\n"
        "_modal_dist_timeout = __import__('datetime').timedelta("
        "minutes=int(os.environ['CONVERSION_DIST_TIMEOUT_MINUTES']))\n"
        "try:\n"
        "    import torch.distributed.constants as _modal_dist_constants\n"
        "    import torch.distributed.distributed_c10d as _modal_dist_c10d\n"
        "    _modal_dist_constants.default_pg_timeout = _modal_dist_timeout\n"
        "    _modal_dist_c10d.default_pg_timeout = _modal_dist_timeout\n"
        "except Exception:\n"
        "    pass\n"
        "_modal_original_new_group = dist.new_group\n"
        "def _modal_new_group_with_timeout(*args, **kwargs):\n"
        "    kwargs.setdefault('timeout', _modal_dist_timeout)\n"
        "    return _modal_original_new_group(*args, **kwargs)\n"
        "dist.new_group = _modal_new_group_with_timeout\n",
    )
    _src = _src.replace(
        'dist.init_process_group(\n        backend="nccl",',
        "dist.init_process_group(\n"
        "        timeout=_modal_dist_timeout,\n"
        '        backend="nccl",',
    )

if os.environ.get("SKIP_RELEASE_RENAME"):
    _src = _src.replace(
        "shutil.move(source_dir, target_dir)",
        "pass  # SKIP_RELEASE_RENAME",
    )
    _src = _src.replace(
        'f.write("release")',
        'f.write("1")  # SKIP_RELEASE_RENAME: keep iter_0000001',
    )
    exec(compile(_src, _UPSTREAM, "exec"))
else:
    exec(compile(_src, _UPSTREAM, "exec"))

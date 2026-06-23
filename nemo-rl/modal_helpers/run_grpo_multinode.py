"""
Helpers to run NemoRL multinode jobs on Modal.

There are two problems that break colocated training across Modal nodes in NemoRL:

- Placement: Colocated VLLM has fractional Ray GPUs (0.5 per actor) so all 16 workers
  could fit in one node's placement group, which leaves the other node empty (in NemoRL, SLURM scheduler ensures each pg goes to a separate node). We force a
  single placement group of 16 whole-GPU bundles and spread them across nodes with Ray's
  SPREAD, whereas the default NemoRL recipe will create one placement group per node. This way gpus_per_node bundles are put into one node, which ensures one rank per GPU.

- Once placement is right, Modal gives every container the same NCCL_HOSTID and same hostname, and NemoRL will copy the driver's env onto every worker. 
Because of this, when NCCL keys on (hostHash, busId) it will find two gpus on the same rank on different nodes (ie. gpu 2 on nodes 0 and 1) are the same device and give a duplicate GPU error. 

To fix this we manually set each worker's NCCL_HOSTID from the physical Ray node its bundle landed on, so ranks on different nodes look distinct.
"""

from __future__ import annotations

import os
import runpy

# Force bundles to spread across nodes. NemoRL defaults to one placement group per node
# (relying on SLURM to keep them apart); on Modal we use a single unified placement group
# so Ray's SPREAD strategy balances whole-GPU bundles across nodes instead of packing them
# onto one.
def _patch_virtual_cluster() -> None:
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster

    if getattr(RayVirtualCluster, "_modal_multinode_patched", False):
        return

    _orig_init_pg = RayVirtualCluster._init_placement_groups

    def _init_placement_groups(self, strategy=None, use_unified_pg=False):
        if len(self._bundle_ct_per_node_list) > 1 and self.use_gpus:
            use_unified_pg = True
        return _orig_init_pg(self, strategy=strategy, use_unified_pg=use_unified_pg)

    RayVirtualCluster._init_placement_groups = _init_placement_groups
    RayVirtualCluster._modal_multinode_patched = True


def _bundle_node_id(placement_group, bundle_index: int) -> str | None:
    """Physical Ray node id hosting a given bundle of a placement group.

    The node id is unique per machine, so it's the right seed for a per-node NCCL
    host hash. Best-effort: returns None if the placement table isn't populated.
    """
    from ray.util.placement_group import placement_group_table

    try:
        table = placement_group_table(placement_group)
        return table.get("bundles_to_node_id", {}).get(bundle_index)
    except Exception:
        return None


# Modal gives every container the same NCCL_HOSTID, so NCCL keys on (hostHash, busId)
# and treats same-busId GPUs on different nodes as one device ("Duplicate GPU detected").
# We override NCCL_HOSTID per physical node.
def _patch_worker_group_hostid() -> None:
    """Give each worker an NCCL_HOSTID keyed to its physical node."""
    from nemo_rl.distributed import worker_groups as wg

    if getattr(wg.RayWorkerBuilder, "_modal_hostid_patched", False):
        return

    _orig_create_worker_async = wg.RayWorkerBuilder.create_worker_async

    def create_worker_async(
        self,
        placement_group,
        placement_group_bundle_index,
        num_gpus,
        bundle_indices=None,
        **extra_options,
    ):
        node_id = _bundle_node_id(placement_group, placement_group_bundle_index)
        if node_id is not None:
            env_vars = extra_options.setdefault("runtime_env", {}).setdefault(
                "env_vars", {}
            )
            env_vars["NCCL_HOSTID"] = f"modal-node-{node_id}"
            print(
                f"[modal] bundle={placement_group_bundle_index} "
                f"NCCL_HOSTID=modal-node-{node_id}",
                flush=True,
            )
        return _orig_create_worker_async(
            self,
            placement_group,
            placement_group_bundle_index,
            num_gpus,
            bundle_indices,
            **extra_options,
        )

    wg.RayWorkerBuilder.create_worker_async = create_worker_async
    wg.RayWorkerBuilder._modal_hostid_patched = True


def main() -> None:
    # Surface NCCL's per-rank hostHash/busId so duplicate-GPU errors are diagnosable.
    # This is set in the driver env, which NeMo-RL copies onto every worker.
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,ENV")
    _patch_virtual_cluster()
    _patch_worker_group_hostid()
    runpy.run_path("examples/run_grpo.py", run_name="__main__")


if __name__ == "__main__":
    main()

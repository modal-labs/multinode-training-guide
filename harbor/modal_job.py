import modal
import harbor
import time
import os
import asyncio

app = modal.App.lookup("vmax-tasks", create_if_missing=True)

MODAL_RATE_LIMIT_CONSTANT = 5
MODAL_RATE_LIMIT_BURST = 150
# Number of sandboxes to launch concurrently
VMAX_N_CONCURRENT = 1000
# Avoid hitting the rate limit in case of timing discrepancies
MARGIN_OF_SAFETY = 1


async def scale_sandboxes():
    async with asyncio.TaskGroup() as tg:
        for i in range(MODAL_RATE_LIMIT_BURST):
            if VMAX_N_CONCURRENT > 0:
                tg.create_task(
                    modal.Sandbox.create(
                        app=app, environment_name=f"vmax-environment-{i + 1}"
                    )
                )
                VMAX_N_CONCURRENT -= 1
        for j in range(VMAX_N_CONCURRENT):
            tg.create_task(
                modal.Sandbox.create(
                    app=app,
                    environment_name=f"vmax-environment-{MODAL_RATE_LIMIT_BURST + j + 1}",
                )
            )
            asyncio.sleep(1 / (MODAL_RATE_LIMIT_CONSTANT - MARGIN_OF_SAFETY))

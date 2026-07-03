"""ComfyUI-NK2E: in-context image editing for Krea 2. Community project, not affiliated with Krea."""
from comfy_api.latest import ComfyExtension, io
from .nk2e_nodes import (
    NK2EInContextModelNode,
    NK2ESetReferenceNode,
    NK2EInContextEditNode,
)


class NK2EExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            NK2EInContextModelNode,     # reload-free: pair with NK2E Set Reference
            NK2ESetReferenceNode,
            NK2EInContextEditNode,      # legacy single node (reloads on ref change)
        ]


async def comfy_entrypoint() -> NK2EExtension:
    return NK2EExtension()

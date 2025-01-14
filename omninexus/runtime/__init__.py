from omninexus.core.logger import omninexus_logger as logger
from omninexus.runtime.impl.docker.docker_runtime import (
    DockerRuntime,
)
from omninexus.runtime.impl.e2b.sandbox import E2BBox
from omninexus.runtime.impl.modal.modal_runtime import ModalRuntime
from omninexus.runtime.impl.remote.remote_runtime import RemoteRuntime
from omninexus.runtime.impl.runloop.runloop_runtime import RunloopRuntime


def get_runtime_cls(name: str):
    # Local imports to avoid circular imports
    if name == 'eventstream' or name == 'docker':
        return DockerRuntime
    elif name == 'e2b':
        return E2BBox
    elif name == 'remote':
        return RemoteRuntime
    elif name == 'modal':
        logger.debug('Using ModalRuntime')
        return ModalRuntime
    elif name == 'runloop':
        return RunloopRuntime
    else:
        raise ValueError(f'Runtime {name} not supported')


__all__ = [
    'E2BBox',
    'RemoteRuntime',
    'ModalRuntime',
    'RunloopRuntime',
    'DockerRuntime',
    'get_runtime_cls',
]

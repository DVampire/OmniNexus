import subprocess
import time
from dataclasses import dataclass

from omninexus.core.logger import omninexus_logger as logger
from omninexus.events.action import Action, IPythonRunCellAction
from omninexus.events.observation import IPythonRunCellObservation
from omninexus.runtime.plugins.jupyter.execute_server import JupyterKernel
from omninexus.runtime.plugins.requirement import Plugin, PluginRequirement
from omninexus.runtime.utils import find_available_tcp_port
from omninexus.utils.shutdown_listener import should_continue


@dataclass
class JupyterRequirement(PluginRequirement):
    name: str = 'jupyter'


class JupyterPlugin(Plugin):
    name: str = 'jupyter'

    async def initialize(self, username: str, kernel_id: str = 'omninexus-default'):
        self.kernel_gateway_port = find_available_tcp_port(40000, 49999)
        self.kernel_id = kernel_id
        self.gateway_process = subprocess.Popen(
            (
                f"su - {username} -s /bin/bash << 'EOF'\n"
                'cd /omninexus/code\n'
                'export POETRY_VIRTUALENVS_PATH=/omninexus/poetry;\n'
                'export PYTHONPATH=/omninexus/code:$PYTHONPATH;\n'
                'export MAMBA_ROOT_PREFIX=/omninexus/micromamba;\n'
                '/omninexus/micromamba/bin/micromamba run -n omninexus '
                'poetry run jupyter kernelgateway '
                '--KernelGatewayApp.ip=0.0.0.0 '
                f'--KernelGatewayApp.port={self.kernel_gateway_port}\n'
                'EOF'
            ),
            stderr=subprocess.STDOUT,
            shell=True,
        )
        # read stdout until the kernel gateway is ready
        output = ''
        while should_continue() and self.gateway_process.stdout is not None:
            line = self.gateway_process.stdout.readline().decode('utf-8')
            output += line
            if 'at' in line:
                break
            time.sleep(1)
            logger.debug('Waiting for jupyter kernel gateway to start...')

        logger.debug(
            f'Jupyter kernel gateway started at port {self.kernel_gateway_port}. Output: {output}'
        )
        _obs = await self.run(
            IPythonRunCellAction(code='import sys; print(sys.executable)')
        )
        self.python_interpreter_path = _obs.content.strip()

    async def _run(self, action: Action) -> IPythonRunCellObservation:
        """Internal method to run a code cell in the jupyter kernel."""
        if not isinstance(action, IPythonRunCellAction):
            raise ValueError(
                f'Jupyter plugin only supports IPythonRunCellAction, but got {action}'
            )

        if not hasattr(self, 'kernel'):
            self.kernel = JupyterKernel(
                f'localhost:{self.kernel_gateway_port}', self.kernel_id
            )

        if not self.kernel.initialized:
            await self.kernel.initialize()
        output = await self.kernel.execute(action.code, timeout=action.timeout)
        return IPythonRunCellObservation(
            content=output,
            code=action.code,
        )

    async def run(self, action: Action) -> IPythonRunCellObservation:
        obs = await self._run(action)
        return obs

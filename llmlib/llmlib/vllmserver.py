from dataclasses import dataclass
import subprocess
import time
import requests
import logging
from typing import Optional
from contextlib import contextmanager
import signal
import atexit
import os

logger = logging.getLogger(__name__)


@dataclass
class VLLMServer:
    cmd: list[str]
    timeout_mins: int = 10

    def __post_init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.port = int(
            [part for part in self.cmd if "--port" in part][0].split("=")[1]
        )
        # Register cleanup handler
        atexit.register(self.stop)

    def start(self) -> None:
        """
        Start the vLLM server and wait for it to be ready.
        Raises RuntimeError if the server process dies unexpectedly.
        Raises TimeoutError if the server fails to start after timeout_mins minutes.
        """
        logger.info("Starting vLLM server with command: %s", " ".join(self.cmd))
        self.process = subprocess.Popen(
            self.cmd, preexec_fn=os.setsid
        )  # Create new process group

        # Wait for server to be ready
        wait_time = 15  # seconds between attempts
        max_attempts = self.timeout_mins * 4

        for attempt in range(max_attempts):
            if not self.is_running():
                raise RuntimeError("Server process died unexpectedly")

            try:
                response = requests.get(f"http://localhost:{self.port}/v1/models")
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                pass

            logger.info(
                "Attempt %d/%d: Server not ready yet, waiting %ds... (%ds elapsed)",
                attempt + 1,
                max_attempts,
                wait_time,
                (attempt + 1) * wait_time,
            )
            time.sleep(wait_time)

        raise TimeoutError(
            f"Server failed to start after {max_attempts} attempts (timeout: {self.timeout_mins} mins)"
        )

    def is_running(self) -> bool:
        """Check if the server process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self):
        """Stop the vLLM server."""
        if self.process is not None and self.is_running():
            logger.info("Stopping vLLM server (PID: %s)...", self.process.pid)
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)  # seconds
            except (subprocess.TimeoutExpired, ProcessLookupError):
                logger.warning("Server did not terminate gracefully, forcing...")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            finally:
                self.process = None
                # Unregister the cleanup handler
                atexit.unregister(self.stop)

    def __enter__(self):
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.stop()


@contextmanager
def spinup_vllm_server(no_op: bool, vllm_command: list[str], timeout_mins: int = 10):
    if no_op:
        yield
        return

    server = VLLMServer(cmd=vllm_command, timeout_mins=timeout_mins)
    try:
        server.start()
        yield server
    finally:
        server.stop()

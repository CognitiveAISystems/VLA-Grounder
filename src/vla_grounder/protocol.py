from __future__ import annotations

import re
from typing import Any

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"</think>\s*(.+)", re.DOTALL)


def extract_evaluation_command(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    if match:
        answer = match.group(1).strip()
        return answer[-150:] if len(answer) >= 150 else answer
    match = THINK_PATTERN.search(text)
    if match:
        answer = match.group(1).strip()
        if answer:
            return answer[-150:] if len(answer) >= 150 else answer
    return text[-100:]


def extract_training_command(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    if match:
        answer = match.group(1).strip()
        return answer[-200:] if len(answer) >= 200 else answer
    match = THINK_PATTERN.search(text)
    if match:
        answer = match.group(1).strip()
        if answer:
            return answer[-200:] if len(answer) >= 200 else answer
    return "stop"


class EnvironmentClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 600_000):
        import zmq

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    def request(self, command: str, data: Any = None) -> Any:
        self.socket.send_pyobj({"command": command, "data": data})
        response = self.socket.recv_pyobj()
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error", "Environment request failed"))
        return response.get("data")

    def close(self) -> None:
        self.socket.close()
        self.context.term()

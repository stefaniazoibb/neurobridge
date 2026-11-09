"""Runs tools on behalf of agents: validates inputs, retries transient failures, records every call."""

from __future__ import annotations

import time
import traceback
import uuid
from typing import Any, Callable, Optional

from pydantic import ValidationError

from neurobridge.agents.tools.registry import REGISTRY, ToolOutput, TransientToolError
from neurobridge.agents.workspace import Workspace
from neurobridge.data.cache import DownloadError
from neurobridge.logging_utils import get_logger, log_event
from neurobridge.provenance import utc_now_iso
from neurobridge.schemas.agents import ExecutionLog, ToolCall, ToolStatus

log = get_logger(__name__)
TRANSIENT = (TransientToolError, DownloadError, ConnectionError, TimeoutError)


class ToolExecutor:
    def __init__(self, workspace: Workspace, exec_log: ExecutionLog, progress: Optional[Callable[[dict], None]] = None, max_attempts: int = 3, backoff_s: float = 1.0):
        self.ws = workspace
        self.log = exec_log
        self.progress = progress
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s

    def _emit(self, **event: Any) -> None:
        self.log.add_event(**event)
        if self.progress:
            try:
                self.progress(event)
            except Exception:  # noqa: BLE001
                pass

    def run(self, tool_name: str, agent: str, **inputs: Any) -> Optional[ToolOutput]:
        tool = REGISTRY.get(tool_name)
        call = ToolCall(call_id=uuid.uuid4().hex[:10], tool=tool_name, agent=agent, inputs={k: v for k, v in inputs.items()})
        self.log.tool_calls.append(call)
        try:
            parsed = tool.input_model(**inputs)
        except ValidationError as exc:
            call.status = ToolStatus.FAILED
            call.error = f"input validation failed: {exc.errors()[:3]}"
            self._emit(kind="tool_failed", tool=tool_name, agent=agent, error=call.error)
            return None
        call.inputs = parsed.model_dump(mode="json")
        call.status = ToolStatus.RUNNING
        call.started_at = utc_now_iso()
        self._emit(kind="tool_started", tool=tool_name, agent=agent, call_id=call.call_id)
        t0 = time.time()
        attempts = self.max_attempts if tool.retryable else 1
        for attempt in range(1, attempts + 1):
            call.attempts = attempt
            try:
                out = tool.fn(self.ws, parsed)
                call.status = ToolStatus.DEGRADED if out.degraded else ToolStatus.SUCCEEDED
                call.output_summary = out.summary
                call.output_ref = ",".join(out.refs) if out.refs else None
                call.limitations = out.limitations
                break
            except TRANSIENT as exc:
                call.error = f"{type(exc).__name__}: {exc}"
                log_event(log, "tool.retry", tool=tool_name, attempt=attempt, error=str(exc))
                if attempt < attempts:
                    self._emit(kind="tool_retry", tool=tool_name, agent=agent, attempt=attempt, error=call.error)
                    time.sleep(self.backoff_s * attempt)
                    continue
                call.status = ToolStatus.FAILED
                out = None
            except Exception as exc:  # noqa: BLE001
                call.status = ToolStatus.FAILED
                call.error = f"{type(exc).__name__}: {exc}"
                log_event(log, "tool.failed", tool=tool_name, error=str(exc), tb=traceback.format_exc()[-1500:])
                out = None
                break
        call.finished_at = utc_now_iso()
        call.duration_s = round(time.time() - t0, 2)
        self._emit(kind="tool_finished" if out else "tool_failed", tool=tool_name, agent=agent, call_id=call.call_id, status=call.status.value, summary=call.output_summary, error=call.error, duration_s=call.duration_s)
        return out

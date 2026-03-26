"""
Full trajectory logger for multi-turn RL training data collection.

Captures complete conversation state per turn: system prompt, messages,
assistant response, tool calls, tool results, and file-level actions.
Output is a JSONL file where each line is one turn.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_serialize(obj: Any, max_str_len: int = 50_000) -> Any:
    """Recursively serialize an object to JSON-safe primitives."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:max_str_len] if len(obj) > max_str_len else obj
    if isinstance(obj, bytes):
        return obj[:max_str_len].decode("utf-8", errors="replace")
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v, max_str_len) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v, max_str_len) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _safe_serialize(v, max_str_len) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)[:max_str_len]


class TrajectoryLogger:
    """Logs full multi-turn trajectories for RL training.

    Each ``log_turn`` call appends one JSON line to ``trajectory.jsonl``
    in the configured output directory.
    """

    def __init__(self, output_dir: str, session_id: Optional[str] = None):
        self.output_dir = output_dir
        self.session_id = session_id or datetime.now().strftime("%Y%m%dT%H%M%S")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = os.path.join(output_dir, "trajectory.jsonl")
        self._turn_counter = 0
        self._start_time = time.time()

    def log_turn(
        self,
        *,
        system_prompt: str,
        messages_before: List[Dict[str, str]],
        assistant_response: Dict[str, Any],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        compiled_user_response: Optional[str] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        files_written: Optional[List[str]] = None,
        files_executed: Optional[List[str]] = None,
        phase: str = "implementation",
        extra: Optional[Dict[str, Any]] = None,
        assistant_thinking: Optional[str] = None,
    ) -> None:
        """Append one turn record to the trajectory log.

        Parameters
        ----------
        system_prompt : str
            The system message active for this turn.
        messages_before : list[dict]
            Conversation history *before* this LLM call (user/assistant messages).
        assistant_response : dict
            ``{"content": str, "tool_calls": list}`` from the LLM.
        tool_calls : list[dict] | None
            Structured tool calls issued by the assistant.
        tool_results : list[dict] | None
            Results from executing each tool call.
        compiled_user_response : str | None
            The compiled user message appended after tool execution.
        token_usage : dict | None
            Token counts (input_tokens, output_tokens, etc.).
        files_written : list[str] | None
            File paths written during this turn.
        files_executed : list[str] | None
            Commands / scripts executed during this turn.
        phase : str
            Current phase label (``"implementation"`` or ``"validation"``).
        extra : dict | None
            Arbitrary extra metadata.
        """
        self._turn_counter += 1
        entry = {
            "turn": self._turn_counter,
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 2),
            "session_id": self.session_id,
            "phase": phase,
            "system_prompt": system_prompt,
            "messages_before": _safe_serialize(messages_before),
            "assistant_response": _safe_serialize(assistant_response),
            "tool_calls": _safe_serialize(tool_calls) if tool_calls else [],
            "tool_results": _safe_serialize(tool_results) if tool_results else [],
            "compiled_user_response": compiled_user_response or "",
            "token_usage": _safe_serialize(token_usage) if token_usage else {},
            "files_written": files_written or [],
            "files_executed": files_executed or [],
            "assistant_thinking": assistant_thinking or "",
        }
        if extra:
            entry["extra"] = _safe_serialize(extra)

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[TrajectoryLogger] write error: {exc}")

    def finalize(self, final_summary: Optional[Dict[str, Any]] = None) -> str:
        """Write a final summary line and return the log path."""
        entry = {
            "turn": -1,
            "type": "summary",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "total_turns": self._turn_counter,
            "total_elapsed_s": round(time.time() - self._start_time, 2),
        }
        if final_summary:
            entry["summary"] = _safe_serialize(final_summary)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[TrajectoryLogger] finalize error: {exc}")
        return self.log_path

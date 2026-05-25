from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class MessagePair:
    user: str
    assistant: str


class ConversationHistory:
    def __init__(self, max_pairs: int = 5) -> None:
        self._sessions: dict[str, deque[MessagePair]] = defaultdict(
            lambda: deque(maxlen=max_pairs)
        )

    def add(self, session_id: str, user: str, assistant: str) -> None:
        self._sessions[session_id].append(MessagePair(user=user, assistant=assistant))

    def format_for_prompt(self, session_id: str) -> str:
        pairs = self._sessions.get(session_id)
        if not pairs:
            return "No prior conversation."
        lines: list[str] = []
        for pair in pairs:
            lines.append(f"User: {pair.user}")
            lines.append(f"Assistant: {pair.assistant}")
        return "\n".join(lines)

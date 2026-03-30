"""Connection management for WebSocket clients and Pipecat processes."""

import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from fastapi import WebSocket

from meetingbaas_pipecat.utils.logger import logger

# File path for persisting meeting details across server restarts
_MEETING_DETAILS_FILE = "/tmp/meeting_details.json"


def _load_meeting_details() -> dict:
    """Load meeting details from disk if the file exists."""
    try:
        if os.path.exists(_MEETING_DETAILS_FILE):
            with open(_MEETING_DETAILS_FILE, "r") as f:
                raw = json.load(f)
            # JSON deserializes tuples as lists; convert back to tuples
            return {k: tuple(v) for k, v in raw.items()}
    except Exception as e:
        logger.warning(f"Could not load meeting details from disk: {e}")
    return {}


def _save_meeting_details(data: dict) -> None:
    """Persist meeting details to disk."""
    try:
        with open(_MEETING_DETAILS_FILE, "w") as f:
            # Tuples are serialized as JSON arrays
            json.dump({k: list(v) for k, v in data.items()}, f)
    except Exception as e:
        logger.warning(f"Could not save meeting details to disk: {e}")


class _PersistentMeetingDetails(dict):
    """A dict subclass that persists to disk on every mutation."""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        _save_meeting_details(self)

    def pop(self, key, *args):
        result = super().pop(key, *args)
        _save_meeting_details(self)
        return result


# Global dictionary to store meeting details for each client
# Loaded from disk on startup so state survives server restarts
MEETING_DETAILS: Dict[
    str, Tuple[str, str, Optional[str], bool, str]
] = _PersistentMeetingDetails(
    _load_meeting_details()
)  # client_id -> (meeting_url, persona_name, meetingbaas_bot_id, enable_tools, streaming_audio_frequency)

# Global dictionary to store Pipecat processes
PIPECAT_PROCESSES: Dict[str, subprocess.Popen] = {}  # client_id -> process


class ConnectionRegistry:
    """Manages WebSocket connections for clients and Pipecat."""

    def __init__(self, logger=logger):
        self.active_connections: Dict[str, WebSocket] = {}
        self.pipecat_connections: Dict[str, WebSocket] = {}
        self.logger = logger

    async def connect(
        self, websocket: WebSocket, client_id: str, is_pipecat: bool = False
    ):
        """Register a new connection."""
        await websocket.accept()
        if is_pipecat:
            self.pipecat_connections[client_id] = websocket
            self.logger.info(f"Pipecat client {client_id} connected")
        else:
            self.active_connections[client_id] = websocket
            self.logger.info(f"Client {client_id} connected")

    async def disconnect(self, client_id: str, is_pipecat: bool = False):
        """Remove a connection and close the websocket."""
        try:
            if is_pipecat:
                if client_id in self.pipecat_connections:
                    websocket = self.pipecat_connections.pop(client_id)
                    try:
                        await websocket.close(code=1000, reason="Bot disconnected")
                    except Exception as e:
                        self.logger.debug(
                            f"Could not close Pipecat WebSocket for {client_id}: {e}"
                        )
                    self.logger.info(f"Pipecat client {client_id} disconnected")
            else:
                if client_id in self.active_connections:
                    websocket = self.active_connections.pop(client_id)
                    try:
                        await websocket.close(code=1000, reason="Bot disconnected")
                    except Exception as e:
                        self.logger.debug(
                            f"Could not close client WebSocket for {client_id}: {e}"
                        )
                    self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.debug(f"Error during disconnect for {client_id}: {e}")

    def get_client(self, client_id: str) -> Optional[WebSocket]:
        """Get a client connection by ID."""
        return self.active_connections.get(client_id)

    def get_pipecat(self, client_id: str) -> Optional[WebSocket]:
        """Get a Pipecat connection by ID."""
        return self.pipecat_connections.get(client_id)


# Create a singleton instance
registry = ConnectionRegistry()

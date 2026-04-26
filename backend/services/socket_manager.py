"""
WebSocket Connection Manager for real-time training metrics.
"""
import sys
from pathlib import Path
from typing import Dict, Set
from fastapi import WebSocket
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.message_history = defaultdict(list)
        self._state: Dict[str, dict] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
        if self.message_history[client_id]:
            for msg in self.message_history[client_id]:
                await websocket.send_json(msg)

    def disconnect(self, client_id: str, websocket: WebSocket | None = None):
        if client_id not in self.active_connections:
            return
        if websocket is not None:
            self.active_connections[client_id].discard(websocket)
        else:
            self.active_connections[client_id].clear()
        if not self.active_connections[client_id]:
            del self.active_connections[client_id]

    def get_state(self, client_id: str) -> dict | None:
        return self._state.get(client_id)

    async def broadcast_json(self, client_id: str, data: dict):
        self._state[client_id] = data
        self.message_history[client_id].append(data)
        if len(self.message_history[client_id]) > 100:
            self.message_history[client_id] = self.message_history[client_id][-100:]
        if client_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_json(data)
                except Exception:
                    disconnected.add(connection)
            for dead in disconnected:
                self.disconnect(client_id, dead)

    async def send_status(self, client_id: str, status: str, message: str = ""):
        await self.broadcast_json(client_id, {"type": "status", "status": status, "message": message})

    async def send_completion(self, client_id: str, success: bool, model_path: str = ""):
        await self.broadcast_json(client_id, {
            "type": "completion",
            "success": success,
            "model_path": model_path,
            "message": "Training completed successfully!" if success else "Training failed"
        })


manager = ConnectionManager()

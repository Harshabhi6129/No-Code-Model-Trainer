# ws_broker.py
import asyncio
from typing import Dict, Any
from fastapi import APIRouter

ws_router = APIRouter()

# --------------------------
# WebSocket Broker State
# --------------------------
# This dictionary stores active WebSocket connections by run_id
connections: Dict[str, set] = {}

# --------------------------
# Register a new connection
# --------------------------
async def register(run_id: str, websocket):
    """
    Register a new websocket for a given run_id.
    """
    if run_id not in connections:
        connections[run_id] = set()
    connections[run_id].add(websocket)
    print(f"[WS] Connected: {run_id} ({len(connections[run_id])} clients)")

# --------------------------
# Unregister a connection
# --------------------------
async def unregister(run_id: str, websocket):
    """
    Remove a websocket from the active set.
    """
    if run_id in connections and websocket in connections[run_id]:
        connections[run_id].remove(websocket)
        print(f"[WS] Disconnected: {run_id} ({len(connections[run_id])} clients left)")
        if not connections[run_id]:  # No more listeners
            del connections[run_id]

# --------------------------
# Publish a message to all clients for a run
# --------------------------
def publish(run_id: str, message: Dict[str, Any]):
    """
    Publish a JSON serializable dict to all WebSockets for a run_id.
    """
    if run_id not in connections:
        return

    # Convert to string for sending
    import json
    msg = json.dumps(message)

    # Broadcast asynchronously
    loop = asyncio.get_event_loop()
    for ws in list(connections[run_id]):
        if ws.application_state.name == "CONNECTED":
            loop.create_task(ws.send_text(msg))
        else:
            # Cleanup if closed
            loop.create_task(unregister(run_id, ws))

"""
WebSocket Connection Manager for real-time training metrics.
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Set
from fastapi import WebSocket

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from collections import defaultdict

class ConnectionManager:
    """
    Manages WebSocket connections for real-time training updates.
    Supports multiple connections per client_id for redundancy.
    """

    def __init__(self):
        # client_id -> Set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # client_id -> List of historical messages (capped)
        self.message_history = defaultdict(list)

    async def connect(self, client_id: str, websocket: WebSocket):
        """Add a new WebSocket connection for a client."""
        await websocket.accept()
        
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        
        self.active_connections[client_id].add(websocket)
        print(f"✅ WebSocket connected for client {client_id}. Total connections: {len(self.active_connections[client_id])}")
        
        # Send history
        if self.message_history[client_id]:
            print(f"🔄 Sending {len(self.message_history[client_id])} historical messages for {client_id}")
            for msg in self.message_history[client_id]:
                await websocket.send_json(msg)

    def disconnect(self, client_id: str, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            
            # Clean up empty sets
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
            
            print(f"❌ WebSocket disconnected for client {client_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error sending personal message: {e}")

    async def broadcast_json(self, client_id: str, data: dict):
        """Broadcast a JSON message to all connections for a client."""
        # Always store in history
        self.message_history[client_id].append(data)
        # Cap history at 100 messages
        if len(self.message_history[client_id]) > 100:
            self.message_history[client_id] = self.message_history[client_id][-100:]

        if client_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    print(f"⚠️ Failed to send to {client_id}: {e}")
                    disconnected.add(connection)
            
            # Cleanup dead connections
            for dead in disconnected:
                self.disconnect(client_id, dead)

    async def send_status(self, client_id: str, status: str, message: str = ""):
        """
        Send a status update to the client.
        """
        await self.broadcast_json(client_id, {
            "type": "status",
            "status": status,
            "message": message
        })

    async def send_completion(self, client_id: str, success: bool, model_path: str = ""):
        """
        Send training completion message.
        """
        await self.broadcast_json(client_id, {
            "type": "completion",
            "success": success,
            "model_path": model_path,
            "message": "Training completed successfully!" if success else "Training failed"
        })

# Global singleton instance
manager = ConnectionManager()

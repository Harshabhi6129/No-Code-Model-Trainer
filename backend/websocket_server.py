import json, asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ws_broker     import subscribe, publish
from chat_agent    import handle_command                         # 🆕

router = APIRouter()

@router.websocket("/ws/train/{run_id}")
async def ws_train(ws: WebSocket, run_id: str):
    await ws.accept()
    consumer = asyncio.create_task(_consumer(ws, run_id))
    producer = asyncio.create_task(_producer(ws, run_id))
    done, _ = await asyncio.wait({consumer, producer}, return_when=asyncio.FIRST_COMPLETED)
    for task in done: task.cancel()

# ── sends runner / chat events → frontend ─────────────────────────
async def _producer(ws: WebSocket, run_id: str):
    async for msg in subscribe(run_id):
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            break

# ── receives chat commands from frontend ──────────────────────────
async def _consumer(ws: WebSocket, run_id: str):
    while True:
        try:
            raw  = await ws.receive_text()
            data = json.loads(raw)

            if data.get("event") == "command":
                responses = handle_command(run_id, data.get("text", ""))
                for resp in responses:
                    await publish(run_id, resp)
            # else: ignore
        except WebSocketDisconnect:
            break

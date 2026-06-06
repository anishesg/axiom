"""
FastAPI server with WebSocket support for real-time EEG streaming.

Provides:
- WebSocket endpoint for real-time EEG events
- REST endpoints for calibration and configuration
- Session management
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from eleven.acquisition import create_acquisition, MuseAcquisition, SimulatedMuse
from eleven.preprocessing import EEGPreprocessor
from eleven.artifacts import ArtifactDetector, ControlSignal, DetectionConfig
from eleven.features import AttentionEstimator

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================

class SessionState(str, Enum):
    """Session states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    CALIBRATING = "calibrating"
    ERROR = "error"


class EEGEvent(BaseModel):
    """Event sent to clients via WebSocket."""
    type: str
    timestamp: float
    data: dict


class SessionConfig(BaseModel):
    """Configuration for EEG session."""
    use_simulation: bool = False
    notch_freq: float = 60.0  # 50 for Europe, 60 for US


class CalibrationRequest(BaseModel):
    """Request to start calibration."""
    intent: str
    duration_seconds: float = 5.0


# ============================================================================
# Session Manager
# ============================================================================

class EEGSession:
    """
    Manages a single EEG streaming session.

    Handles:
    - Muse connection
    - Signal processing pipeline
    - Event generation
    """

    def __init__(self, config: SessionConfig):
        self.config = config
        self.state = SessionState.DISCONNECTED

        # Components
        self.muse: Optional[MuseAcquisition | SimulatedMuse] = None
        self.preprocessor: Optional[EEGPreprocessor] = None
        self.artifact_detector: Optional[ArtifactDetector] = None
        self.attention_estimator: Optional[AttentionEstimator] = None

        # Event queue for WebSocket clients
        self.event_queue: asyncio.Queue[EEGEvent] = asyncio.Queue()

        # Processing task
        self._process_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def connect(self) -> bool:
        """Connect to Muse headband."""
        if self.state != SessionState.DISCONNECTED:
            return False

        self.state = SessionState.CONNECTING

        try:
            # Create acquisition
            self.muse = create_acquisition(use_simulation=self.config.use_simulation)

            # Connect in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self.muse.connect)

            if not success:
                self.state = SessionState.ERROR
                return False

            # Initialize processing pipeline
            sample_rate = self.muse.get_sample_rate()
            self.preprocessor = EEGPreprocessor(
                sample_rate=sample_rate,
                notch_freq=self.config.notch_freq,
            )
            self.artifact_detector = ArtifactDetector()
            self.attention_estimator = AttentionEstimator(sample_rate)

            self.state = SessionState.CONNECTED
            await self._emit_event("connection", {"status": "connected"})

            return True

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.state = SessionState.ERROR
            return False

    async def start_streaming(self):
        """Start EEG streaming and processing."""
        if self.state != SessionState.CONNECTED:
            return

        self._stop_event.clear()

        # Start Muse stream
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.muse.start_stream)

        self.state = SessionState.STREAMING
        await self._emit_event("streaming", {"status": "started"})

        # Start processing task
        self._process_task = asyncio.create_task(self._process_loop())

    async def _process_loop(self):
        """Main processing loop."""
        while not self._stop_event.is_set():
            try:
                # Get data from Muse
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None, lambda: self.muse.get_data(64)
                )

                if data is None or data.shape[1] == 0:
                    await asyncio.sleep(0.05)
                    continue

                # Process through pipeline
                windows = self.preprocessor.process(data)

                for window in windows:
                    timestamp = time.time()

                    # Check for artifacts (control signals)
                    signal = self.artifact_detector.process(window, timestamp)
                    if signal is not None and signal != ControlSignal.NONE:
                        await self._emit_event("control_signal", {
                            "signal": signal.value,
                        })

                    # Compute attention metrics
                    attention = self.attention_estimator.estimate(window)
                    await self._emit_event("attention", attention)

                # Small delay to prevent busy-waiting
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing error: {e}")
                await asyncio.sleep(0.1)

    async def stop_streaming(self):
        """Stop EEG streaming."""
        self._stop_event.set()

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None

        if self.muse:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.muse.stop_stream)

        self.state = SessionState.CONNECTED
        await self._emit_event("streaming", {"status": "stopped"})

    async def disconnect(self):
        """Disconnect from Muse."""
        await self.stop_streaming()

        if self.muse:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.muse.disconnect)
            self.muse = None

        self.state = SessionState.DISCONNECTED
        await self._emit_event("connection", {"status": "disconnected"})

    async def _emit_event(self, event_type: str, data: dict):
        """Add event to queue for WebSocket clients."""
        event = EEGEvent(
            type=event_type,
            timestamp=time.time(),
            data=data,
        )
        await self.event_queue.put(event)

    async def get_event(self) -> EEGEvent:
        """Get next event from queue."""
        return await self.event_queue.get()


# ============================================================================
# Global Session (single user for now)
# ============================================================================

_session: Optional[EEGSession] = None


def get_session() -> Optional[EEGSession]:
    return _session


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Eleven server starting...")
    yield
    logger.info("Eleven server shutting down...")

    # Cleanup session
    global _session
    if _session:
        await _session.disconnect()
        _session = None


app = FastAPI(
    title="Eleven",
    description="EEG-based communication platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/session/state")
async def get_session_state():
    """Get current session state."""
    session = get_session()
    if session is None:
        return {"state": SessionState.DISCONNECTED.value}
    return {"state": session.state.value}


@app.post("/session/connect")
async def connect_session(config: SessionConfig):
    """Connect to Muse headband."""
    global _session

    if _session and _session.state != SessionState.DISCONNECTED:
        raise HTTPException(400, "Session already active")

    _session = EEGSession(config)
    success = await _session.connect()

    if not success:
        raise HTTPException(500, "Failed to connect to Muse")

    return {"status": "connected"}


@app.post("/session/start")
async def start_streaming():
    """Start EEG streaming."""
    session = get_session()
    if session is None or session.state != SessionState.CONNECTED:
        raise HTTPException(400, "No connected session")

    await session.start_streaming()
    return {"status": "streaming"}


@app.post("/session/stop")
async def stop_streaming():
    """Stop EEG streaming."""
    session = get_session()
    if session is None:
        raise HTTPException(400, "No session")

    await session.stop_streaming()
    return {"status": "stopped"}


@app.post("/session/disconnect")
async def disconnect_session():
    """Disconnect from Muse."""
    global _session

    if _session:
        await _session.disconnect()
        _session = None

    return {"status": "disconnected"}


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time EEG events.

    Events sent to client:
    - connection: {status: "connected" | "disconnected"}
    - streaming: {status: "started" | "stopped"}
    - control_signal: {signal: "single_blink" | "double_blink" | ...}
    - attention: {focus: 0-1, relaxation: 0-1, engagement: 0-1}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    session = get_session()

    try:
        while True:
            if session is None:
                session = get_session()
                await asyncio.sleep(0.1)
                continue

            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(
                    session.get_event(),
                    timeout=1.0
                )
                await websocket.send_json(event.model_dump())

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": time.time(),
                    "data": {"state": session.state.value}
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run the server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    uvicorn.run(
        "eleven.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()

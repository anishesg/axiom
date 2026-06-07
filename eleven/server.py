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
import os
import time

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from typing import Optional, Set
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import httpx

# Optional: Anthropic for Claude AI
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from pathlib import Path

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes

from eleven.acquisition import create_acquisition, MuseAcquisition, SimulatedMuse
from eleven.preprocessing import EEGPreprocessor
from eleven.artifacts import ArtifactDetector, ControlSignal, DetectionConfig, CalibrationSession, EnhancedCalibrationSession
from eleven.features import AttentionEstimator, MultiScaleFeatureExtractor
from eleven.user_profile import UserProfile, ProfileManager
from eleven.vq_encoder import EEGTokenizer, RVQConfig


def filter_eeg_channel(data: np.ndarray, sample_rate: int = 256) -> np.ndarray:
    """Apply standard EEG preprocessing to a single channel.

    From research branch - applies detrend, bandpass (1-50Hz), and 60Hz notch filter.
    """
    filtered = data.copy()
    if len(filtered) < 12:
        return filtered
    DataFilter.detrend(filtered, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(
        filtered, sample_rate,
        1.0, 50.0, 4,
        FilterTypes.BUTTERWORTH.value, 0.0,
    )
    DataFilter.remove_environmental_noise(
        filtered, sample_rate, NoiseTypes.SIXTY.value,
    )
    return filtered

# Default profiles directory
PROFILES_DIR = Path.home() / ".eleven" / "profiles"

logger = logging.getLogger(__name__)

# Connection lock to prevent race conditions
_connection_lock = asyncio.Lock()

# Connection timeout in seconds
CONNECTION_TIMEOUT = 30.0


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


class ConnectionManager:
    """Manages WebSocket connections and broadcasts events to all clients."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        if not self.active_connections:
            return
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)


# Global connection manager for WebSocket broadcast
ws_manager = ConnectionManager()


class CalibrationStep(str, Enum):
    """Calibration steps."""
    REST = "rest"
    NATURAL_BLINKS = "natural_blinks"
    DELIBERATE_BLINKS = "deliberate_blinks"
    JAW_CLENCHES = "jaw_clenches"


class CalibrationStepRequest(BaseModel):
    """Request to start a calibration step."""
    step: CalibrationStep
    duration_seconds: float = 5.0


class CalibrationStatus(BaseModel):
    """Current calibration status."""
    active: bool
    current_step: Optional[str] = None
    progress: float = 0.0
    samples_collected: int = 0
    completed_steps: list[str] = []


class ProfileResponse(BaseModel):
    """User profile response."""
    user_id: str
    created_at: str
    calibration_complete: bool
    baseline_collected: bool
    intents_trained: list[str] = []
    total_sessions: int = 0


class BaselineStartRequest(BaseModel):
    """Request to start baseline collection."""
    duration_seconds: float = 120.0  # 2 minutes default


class EnhancedCalibrationStepRequest(BaseModel):
    """Request for enhanced calibration step."""
    step_id: str
    duration_seconds: Optional[float] = None  # Use default from step config


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

        # Calibration
        self.calibration: Optional[CalibrationSession] = None
        self._calibration_step: Optional[CalibrationStep] = None
        self._calibration_start_time: float = 0.0
        self._calibration_duration: float = 0.0
        self._calibration_samples: int = 0
        self._completed_steps: list[str] = []

    async def connect(self) -> bool:
        """Connect to Muse headband."""
        if self.state != SessionState.DISCONNECTED:
            logger.warning(f"Cannot connect: current state is {self.state.value}")
            return False

        self.state = SessionState.CONNECTING
        logger.info(f"Starting connection (simulation={self.config.use_simulation})...")

        try:
            # Create acquisition
            self.muse = create_acquisition(use_simulation=self.config.use_simulation)
            logger.info("Acquisition created, starting BLE scan...")

            # Connect in thread pool with timeout (blocking operation)
            loop = asyncio.get_event_loop()
            try:
                success = await asyncio.wait_for(
                    loop.run_in_executor(None, self.muse.connect),
                    timeout=CONNECTION_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"Connection timed out after {CONNECTION_TIMEOUT}s")
                self.state = SessionState.ERROR
                # Clean up the muse object
                if self.muse:
                    try:
                        self.muse.disconnect()
                    except Exception:
                        pass
                    self.muse = None
                return False

            if not success:
                logger.error("Muse connection returned False")
                self.state = SessionState.ERROR
                return False

            logger.info("BLE connection successful, initializing pipeline...")

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
            logger.info("Connection complete - state: CONNECTED")

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

                    # Feed window to enhanced calibration if active
                    if hasattr(self, '_enhanced_calibration'):
                        self._enhanced_calibration.add_window(window, timestamp)

                        # Emit progress events (throttled to ~1/sec)
                        enhanced_cal = self._enhanced_calibration
                        if enhanced_cal._is_collecting and enhanced_cal.current_step:
                            step = enhanced_cal.current_step
                            step_id = step["id"]
                            samples = len(enhanced_cal._collected_windows.get(step_id, []))
                            # Only emit every 4 samples (~1/sec at 4 windows/sec)
                            if samples % 4 == 0:
                                elapsed = timestamp - (enhanced_cal._step_start_time or timestamp)
                                await self._emit_event("calibration_progress", {
                                    "step_id": step_id,
                                    "samples_collected": samples,
                                    "elapsed_seconds": round(elapsed, 1),
                                })

                    # Check for artifacts (control signals)
                    signal = self.artifact_detector.process(window, timestamp)
                    if signal is not None and signal != ControlSignal.NONE:
                        await self._emit_event("control_signal", {
                            "signal": signal.value,
                        })

                    # Compute attention metrics
                    attention = self.attention_estimator.estimate(window)
                    await self._emit_event("attention", attention)

                    # Emit EEG data for visualization (~4Hz, every window)
                    # Skip additional filtering - preprocessing already handles this
                    eeg_data = window[:4]  # Shape: (4, samples)

                    # Protect against any stray NaN/Inf values
                    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)

                    # Downsample by 4 for efficiency (256Hz → 64Hz display)
                    downsampled = eeg_data[:, ::4]

                    # Send downsampled data
                    await self._emit_event("eeg", {
                        "channels": ["TP9", "AF7", "AF8", "TP10"],
                        "data": downsampled.tolist(),
                        "sampleRate": 64,
                        "timestamp": timestamp,
                    })

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

    # Calibration methods
    async def start_calibration(self):
        """Start a new calibration session."""
        if self.state not in (SessionState.CONNECTED, SessionState.STREAMING):
            return False

        self.calibration = CalibrationSession()
        self._completed_steps = []
        self.state = SessionState.CALIBRATING
        await self._emit_event("calibration", {
            "status": "started",
            "message": "Calibration session started"
        })
        return True

    async def start_calibration_step(self, step: CalibrationStep, duration: float):
        """Start collecting data for a calibration step."""
        if self.calibration is None:
            return False

        self._calibration_step = step
        self._calibration_start_time = time.time()
        self._calibration_duration = duration
        self._calibration_samples = 0

        self.calibration.start_step(step.value)

        await self._emit_event("calibration_step", {
            "status": "started",
            "step": step.value,
            "duration": duration,
            "message": self._get_step_instruction(step)
        })
        return True

    def _get_step_instruction(self, step: CalibrationStep) -> str:
        """Get user instruction for calibration step."""
        instructions = {
            CalibrationStep.REST: "Relax and keep your eyes open. Try not to blink.",
            CalibrationStep.NATURAL_BLINKS: "Blink naturally as you normally would.",
            CalibrationStep.DELIBERATE_BLINKS: "Blink firmly and deliberately when you see the prompt.",
            CalibrationStep.JAW_CLENCHES: "Clench your jaw firmly when you see the prompt.",
        }
        return instructions.get(step, "Follow the on-screen instructions.")

    async def end_calibration_step(self):
        """End the current calibration step."""
        if self.calibration is None or self._calibration_step is None:
            return False

        self.calibration.end_step(self._calibration_step.value)
        self._completed_steps.append(self._calibration_step.value)

        await self._emit_event("calibration_step", {
            "status": "completed",
            "step": self._calibration_step.value,
            "samples": self._calibration_samples
        })

        self._calibration_step = None
        return True

    async def complete_calibration(self) -> Optional[DetectionConfig]:
        """Complete calibration and get personalized thresholds."""
        if self.calibration is None:
            return None

        config = self.calibration.get_config()

        # Apply the new thresholds
        if self.artifact_detector:
            self.artifact_detector = ArtifactDetector(config)

        await self._emit_event("calibration", {
            "status": "completed",
            "thresholds": {
                "blink_threshold": config.blink_threshold,
                "clench_threshold": config.clench_threshold,
            }
        })

        self.state = SessionState.CONNECTED
        return config

    def get_calibration_status(self) -> dict:
        """Get current calibration status."""
        if self.calibration is None:
            return {
                "active": False,
                "current_step": None,
                "progress": 0.0,
                "samples_collected": 0,
                "completed_steps": []
            }

        progress = 0.0
        if self._calibration_step and self._calibration_duration > 0:
            elapsed = time.time() - self._calibration_start_time
            progress = min(1.0, elapsed / self._calibration_duration)

        return {
            "active": True,
            "current_step": self._calibration_step.value if self._calibration_step else None,
            "progress": progress,
            "samples_collected": self._calibration_samples,
            "completed_steps": self._completed_steps
        }

    def add_calibration_sample(self, amplitude: float):
        """Add a sample during calibration."""
        if self.calibration and self._calibration_step:
            self.calibration.add_sample(amplitude)
            self._calibration_samples += 1


# ============================================================================
# Global Session (single user for now)
# ============================================================================

_session: Optional[EEGSession] = None
_profile_manager: Optional[ProfileManager] = None
_current_user_id: str = "default"


def get_session() -> Optional[EEGSession]:
    return _session


def get_profile_manager() -> ProfileManager:
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager(PROFILES_DIR)
    return _profile_manager


# ============================================================================
# WebSocket Event Broadcasting
# ============================================================================

async def broadcast_events():
    """Background task to broadcast events to all WebSocket clients."""
    while True:
        session = get_session()
        if session is None:
            await asyncio.sleep(0.1)
            continue

        try:
            event = await asyncio.wait_for(
                session.get_event(),
                timeout=1.0
            )
            await ws_manager.broadcast(event.model_dump())
        except asyncio.TimeoutError:
            # Send heartbeat to all clients
            if ws_manager.active_connections:
                await ws_manager.broadcast({
                    "type": "heartbeat",
                    "timestamp": time.time(),
                    "data": {"state": session.state.value}
                })
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(0.1)


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Eleven server starting...")

    # Start the event broadcast task
    broadcast_task = asyncio.create_task(broadcast_events())
    logger.info("WebSocket broadcast task started")

    yield

    logger.info("Eleven server shutting down...")

    # Cancel broadcast task
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass

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
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:5177"],
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

    logger.info(f"Connect request received (simulation={config.use_simulation})")

    # Use lock to prevent concurrent connection attempts
    if _connection_lock.locked():
        logger.warning("Connection already in progress, rejecting duplicate request")
        raise HTTPException(409, "Connection already in progress")

    async with _connection_lock:
        # Only reject if actively streaming or calibrating (states we shouldn't interrupt)
        if _session and _session.state in (SessionState.STREAMING, SessionState.CALIBRATING):
            logger.warning(f"Rejecting connect: session state is {_session.state.value}")
            raise HTTPException(400, "Session already active - streaming or calibrating")

        # Clean up any existing session before creating a new one
        if _session:
            logger.info(f"Cleaning up existing session (state={_session.state.value})")
            try:
                await _session.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting existing session: {e}")
            _session = None

        # Force cleanup any stale BrainFlow sessions that may have been left behind
        # from interrupted connections (e.g., timeout during BLE discovery)
        MuseAcquisition.force_cleanup_all()

        logger.info("Creating new session...")
        _session = EEGSession(config)
        success = await _session.connect()

        if not success:
            # Clean up failed session
            logger.error("Connection failed, cleaning up session")
            _session = None
            raise HTTPException(500, "Failed to connect to Muse - check if device is powered on and nearby")

        logger.info("Connection successful!")
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
# Calibration Endpoints
# ============================================================================

@app.post("/calibration/start")
async def start_calibration():
    """Start a new calibration session."""
    session = get_session()
    if session is None:
        raise HTTPException(400, "No session - connect first")

    if session.state not in (SessionState.CONNECTED, SessionState.STREAMING):
        raise HTTPException(400, f"Cannot calibrate in state: {session.state.value}")

    success = await session.start_calibration()
    if not success:
        raise HTTPException(500, "Failed to start calibration")

    return {"status": "calibrating"}


@app.post("/calibration/step")
async def start_calibration_step(request: CalibrationStepRequest):
    """Start a specific calibration step."""
    session = get_session()
    if session is None or session.calibration is None:
        raise HTTPException(400, "No active calibration session")

    success = await session.start_calibration_step(request.step, request.duration_seconds)
    if not success:
        raise HTTPException(500, "Failed to start calibration step")

    return {
        "status": "collecting",
        "step": request.step.value,
        "duration": request.duration_seconds
    }


@app.post("/calibration/end-step")
async def end_calibration_step():
    """End the current calibration step."""
    session = get_session()
    if session is None or session.calibration is None:
        raise HTTPException(400, "No active calibration session")

    success = await session.end_calibration_step()
    if not success:
        raise HTTPException(400, "No active calibration step")

    return {"status": "step_completed"}


@app.post("/calibration/complete")
async def complete_calibration():
    """Complete calibration and apply personalized thresholds."""
    session = get_session()
    if session is None or session.calibration is None:
        raise HTTPException(400, "No active calibration session")

    config = await session.complete_calibration()
    if config is None:
        raise HTTPException(500, "Failed to complete calibration")

    return {
        "status": "completed",
        "thresholds": {
            "blink_threshold": config.blink_threshold,
            "clench_threshold": config.clench_threshold,
        }
    }


@app.get("/calibration/status")
async def get_calibration_status():
    """Get current calibration status."""
    session = get_session()
    if session is None:
        return CalibrationStatus(active=False)

    status = session.get_calibration_status()
    return CalibrationStatus(**status)


# ============================================================================
# Profile Endpoints
# ============================================================================

@app.get("/profiles")
async def list_profiles():
    """List all available user profiles."""
    manager = get_profile_manager()
    profiles = manager.list_profiles()
    return {"profiles": profiles}


@app.post("/profiles/{user_id}")
async def create_profile(user_id: str):
    """Create a new user profile."""
    manager = get_profile_manager()
    try:
        profile = manager.create_profile(user_id)
        return ProfileResponse(
            user_id=profile.user_id,
            created_at=profile.created_at,
            calibration_complete=profile.progress.is_complete(),
            baseline_collected=profile.progress.baseline_collected,
            intents_trained=profile.intent_names,
            total_sessions=profile.total_sessions,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/profiles/{user_id}")
async def get_profile(user_id: str):
    """Get a user profile."""
    manager = get_profile_manager()
    try:
        profile = manager.load_profile(user_id)
        return ProfileResponse(
            user_id=profile.user_id,
            created_at=profile.created_at,
            calibration_complete=profile.progress.is_complete(),
            baseline_collected=profile.progress.baseline_collected,
            intents_trained=profile.intent_names,
            total_sessions=profile.total_sessions,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.delete("/profiles/{user_id}")
async def delete_profile(user_id: str):
    """Delete a user profile."""
    manager = get_profile_manager()
    manager.delete_profile(user_id)
    return {"status": "deleted", "user_id": user_id}


@app.post("/profiles/{user_id}/activate")
async def activate_profile(user_id: str):
    """Set the active profile for the current session."""
    global _current_user_id
    manager = get_profile_manager()

    try:
        manager.set_current(user_id)
        _current_user_id = user_id
        return {"status": "activated", "user_id": user_id}
    except ValueError as e:
        # Profile doesn't exist, create it
        manager.create_profile(user_id)
        manager.set_current(user_id)
        _current_user_id = user_id
        return {"status": "created_and_activated", "user_id": user_id}


# ============================================================================
# Enhanced Calibration Endpoints (VQ/RVQ)
# ============================================================================

@app.post("/calibration/enhanced/start/{user_id}")
async def start_enhanced_calibration(user_id: str):
    """
    Start an enhanced calibration session with RVQ training.

    This creates an EnhancedCalibrationSession that includes:
    - Baseline collection
    - State calibration (focus, relax, neutral)
    - Action calibration (blinks, clenches)
    - RVQ codebook training
    """
    session = get_session()
    if session is None:
        raise HTTPException(400, "No EEG session - connect first")

    if session.state not in (SessionState.CONNECTED, SessionState.STREAMING):
        raise HTTPException(400, f"Cannot calibrate in state: {session.state.value}")

    # Create enhanced calibration session
    enhanced_cal = EnhancedCalibrationSession(user_id=user_id)

    # Store on session for access in other endpoints
    session._enhanced_calibration = enhanced_cal

    return {
        "status": "started",
        "user_id": user_id,
        "steps": [
            {"id": s["id"], "name": s["name"], "duration": s["duration"]}
            for s in EnhancedCalibrationSession.STEPS
        ]
    }


@app.post("/calibration/enhanced/step")
async def start_enhanced_calibration_step(request: EnhancedCalibrationStepRequest):
    """Start an enhanced calibration step."""
    session = get_session()
    if session is None or not hasattr(session, '_enhanced_calibration'):
        raise HTTPException(400, "No enhanced calibration session active")

    enhanced_cal = session._enhanced_calibration
    success = enhanced_cal.start_step(request.step_id)

    if not success:
        raise HTTPException(400, f"Invalid step: {request.step_id}")

    step = enhanced_cal.current_step
    return {
        "status": "started",
        "step_id": request.step_id,
        "name": step["name"] if step else "",
        "duration": step["duration"] if step else 0,
        "instruction": step["instruction"] if step else "",
    }


@app.post("/calibration/enhanced/end-step")
async def end_enhanced_calibration_step():
    """End the current enhanced calibration step."""
    session = get_session()
    if session is None or not hasattr(session, '_enhanced_calibration'):
        raise HTTPException(400, "No enhanced calibration session active")

    enhanced_cal = session._enhanced_calibration
    results = enhanced_cal.end_step()

    return {"status": "completed", "results": results}


@app.get("/calibration/enhanced/readiness")
async def get_calibration_readiness():
    """
    Check if enough data has been collected for training.

    Returns readiness status, collected states, and recommendations.
    """
    session = get_session()
    if session is None or not hasattr(session, '_enhanced_calibration'):
        raise HTTPException(400, detail={
            "code": "NO_SESSION",
            "message": "No enhanced calibration session active",
            "recoverable": True,
            "action": "Start enhanced calibration first"
        })

    return session._enhanced_calibration.get_training_readiness()


@app.post("/calibration/enhanced/train")
async def train_vq_tokenizer(use_rvq: bool = True):
    """
    Train the VQ/RVQ tokenizer on collected calibration data.

    Args:
        use_rvq: Whether to use Residual VQ (recommended, default True)
    """
    session = get_session()
    if session is None or not hasattr(session, '_enhanced_calibration'):
        raise HTTPException(400, detail={
            "code": "NO_SESSION",
            "message": "No enhanced calibration session active",
            "recoverable": True,
            "action": "Call /calibration/enhanced/start/{user_id} first"
        })

    enhanced_cal = session._enhanced_calibration

    # Check readiness before training
    readiness = enhanced_cal.get_training_readiness()
    if not readiness["ready"]:
        raise HTTPException(400, detail={
            "code": "INSUFFICIENT_DATA",
            "message": "No training data collected",
            "recoverable": False,
            "collected": readiness["collected_states"],
            "missing": readiness["missing_states"],
            "action": readiness["recommendation"]
        })

    # Train the tokenizer
    results = enhanced_cal.train_tokenizer(use_rvq=use_rvq)

    if "error" in results:
        raise HTTPException(400, detail={
            "code": "TRAINING_FAILED",
            "message": results["error"],
            "recoverable": False,
            "action": "Restart calibration and complete state steps"
        })

    return {"status": "trained", "results": results}


@app.post("/calibration/enhanced/complete")
async def complete_enhanced_calibration():
    """Complete enhanced calibration and save the profile."""
    session = get_session()
    if session is None or not hasattr(session, '_enhanced_calibration'):
        raise HTTPException(400, "No enhanced calibration session active")

    enhanced_cal = session._enhanced_calibration

    # Update thresholds
    enhanced_cal.update_thresholds()

    # Save results
    profile_dir = PROFILES_DIR / enhanced_cal.user_id
    enhanced_cal.save_results(profile_dir)

    # Get detection config and apply to session
    config = enhanced_cal.get_detection_config()
    if session.artifact_detector:
        session.artifact_detector = ArtifactDetector(config)

    # Clean up
    delattr(session, '_enhanced_calibration')

    return {
        "status": "completed",
        "user_id": enhanced_cal.user_id,
        "thresholds": {
            "blink_threshold": config.blink_threshold,
            "clench_threshold": config.clench_threshold,
        }
    }


@app.get("/calibration/enhanced/steps")
async def get_enhanced_calibration_steps():
    """Get all enhanced calibration steps."""
    return {
        "steps": [
            {"id": s["id"], "name": s["name"], "duration": s["duration"], "instruction": s["instruction"]}
            for s in EnhancedCalibrationSession.STEPS
        ]
    }


# ============================================================================
# Conversation / AI Response Endpoint
# ============================================================================

class ConversationMessage(BaseModel):
    """A message in the conversation."""
    sender: str  # 'speaker' or 'eeg_user'
    text: str


class CognitiveState(BaseModel):
    """Cognitive state from EEG analysis."""
    engagement: float = 0.5
    focus: float = 0.5
    relaxation: float = 0.5
    cognitive_load: float = 0.5
    valence: float = 0.5


class GenerateResponseRequest(BaseModel):
    """Request to generate an AI response based on EEG cognitive state."""
    speaker_message: str
    cognitive_state: CognitiveState
    conversation_history: list[ConversationMessage] = []
    eeg_user_name: str = "User"


def detect_emotion(state: CognitiveState) -> str:
    """Map cognitive state to emotion label."""
    v = state.valence
    e = state.engagement
    r = state.relaxation
    c = state.cognitive_load

    # High valence states
    if v > 0.6:
        if e > 0.5:
            return "Happy"
        elif r > 0.6:
            return "Calm"
        else:
            return "Content"

    # Low valence states
    elif v < 0.4:
        if e > 0.6 and r < 0.4:
            return "Frustrated"
        elif c > 0.6:
            return "Stressed"
        elif r > 0.5:
            return "Sad"
        else:
            return "Uncomfortable"

    # Neutral valence
    else:
        if e > 0.5:
            return "Attentive"
        elif r > 0.7:
            return "Tired"
        elif c > 0.5:
            return "Thinking"
        else:
            return "Neutral"


def generate_response_from_state(
    speaker_message: str,
    emotion: str,
    state: CognitiveState,
    history: list[ConversationMessage],
    user_name: str
) -> str:
    """Generate a contextual response based on detected emotion and conversation."""

    # Response templates based on emotion
    responses = {
        "Happy": [
            "I'm feeling good right now.",
            "Things are going well, thank you for asking.",
            "I'm in a positive mood today.",
            "Yes, I'm doing great!",
        ],
        "Calm": [
            "I'm feeling peaceful and relaxed.",
            "Everything is fine, I'm comfortable.",
            "I'm at ease right now.",
            "I feel calm and content.",
        ],
        "Content": [
            "I'm doing alright.",
            "Things are okay.",
            "I'm satisfied with how things are.",
            "No complaints here.",
        ],
        "Frustrated": [
            "I'm feeling a bit frustrated.",
            "Something is bothering me.",
            "I'm not entirely happy right now.",
            "This is challenging for me.",
        ],
        "Stressed": [
            "I'm feeling some pressure right now.",
            "Things feel overwhelming.",
            "I could use some help.",
            "I'm stressed about something.",
        ],
        "Sad": [
            "I'm feeling down.",
            "Things could be better.",
            "I'm not in the best mood.",
            "I feel a bit sad.",
        ],
        "Uncomfortable": [
            "I'm not feeling great.",
            "Something doesn't feel right.",
            "I'm a bit uncomfortable.",
            "I'd like things to change.",
        ],
        "Attentive": [
            "I'm listening carefully.",
            "You have my full attention.",
            "I'm focused on what you're saying.",
            "I understand, please continue.",
        ],
        "Tired": [
            "I'm feeling tired.",
            "I could use some rest.",
            "My energy is low right now.",
            "I'm a bit drowsy.",
        ],
        "Thinking": [
            "I'm processing that.",
            "Let me think about it.",
            "I'm considering what you said.",
            "I need a moment to think.",
        ],
        "Neutral": [
            "I'm doing okay.",
            "Nothing particular to report.",
            "Things are normal.",
            "I'm fine.",
        ],
    }

    # Context-aware response selection
    msg_lower = speaker_message.lower()

    # Check for specific questions
    if any(q in msg_lower for q in ["how are you", "how do you feel", "how's it going", "are you okay"]):
        return responses.get(emotion, responses["Neutral"])[0]

    if any(q in msg_lower for q in ["need anything", "can i help", "want something"]):
        if emotion in ["Stressed", "Frustrated", "Uncomfortable"]:
            return "Yes, I could use some help."
        elif emotion == "Tired":
            return "I'd like to rest, please."
        else:
            return "I'm okay for now, thank you."

    if any(q in msg_lower for q in ["yes or no", "do you want", "would you like", "should i"]):
        if state.valence > 0.5:
            return "Yes, please."
        else:
            return "No, thank you."

    if any(q in msg_lower for q in ["understand", "got it", "makes sense"]):
        if state.engagement > 0.5:
            return "Yes, I understand."
        else:
            return "Could you explain more?"

    # Default: return emotion-based response
    import random
    options = responses.get(emotion, responses["Neutral"])
    return random.choice(options)


@app.post("/api/generate-response")
async def generate_response(request: GenerateResponseRequest):
    """
    Generate an AI response based on EEG cognitive state using Claude AI.

    This endpoint analyzes the cognitive state from EEG signals and generates
    a contextual response that the EEG user might want to communicate.
    """
    # Detect emotion from cognitive state
    emotion = detect_emotion(request.cognitive_state)

    # Check if Claude is available
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if ANTHROPIC_AVAILABLE and api_key:
        # Use Claude AI for response generation
        try:
            client = anthropic.Anthropic(api_key=api_key)

            # Build conversation context
            conversation_context = ""
            for msg in request.conversation_history[-5:]:  # Last 5 messages
                role = "Speaker" if msg.sender == "speaker" else request.eeg_user_name
                conversation_context += f"{role}: {msg.text}\n"

            prompt = f"""You are helping a person with disabilities communicate through brain-computer interface (EEG).
Based on their current cognitive/emotional state detected from EEG signals, generate a natural response they might want to say.

Current EEG-detected state for {request.eeg_user_name}:
- Emotional state: {emotion}
- Engagement level: {request.cognitive_state.engagement:.0%}
- Focus level: {request.cognitive_state.focus:.0%}
- Relaxation level: {request.cognitive_state.relaxation:.0%}
- Emotional valence (positive/negative): {request.cognitive_state.valence:.0%}

Recent conversation:
{conversation_context}
Speaker just said: "{request.speaker_message}"

Generate a brief, natural response (1-2 sentences) that {request.eeg_user_name} would likely want to say based on their detected emotional state.
The response should feel authentic and match the emotional tone detected.
Only output the response text, nothing else."""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text.strip()
            logger.info(f"Claude generated response: {response_text}")

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            # Fallback to rule-based
            response_text = generate_response_from_state(
                speaker_message=request.speaker_message,
                emotion=emotion,
                state=request.cognitive_state,
                history=request.conversation_history,
                user_name=request.eeg_user_name
            )
    else:
        # Fallback to rule-based response
        response_text = generate_response_from_state(
            speaker_message=request.speaker_message,
            emotion=emotion,
            state=request.cognitive_state,
            history=request.conversation_history,
            user_name=request.eeg_user_name
        )

    return {
        "response": response_text,
        "emotion": emotion,
        "cognitive_state": {
            "engagement": request.cognitive_state.engagement,
            "focus": request.cognitive_state.focus,
            "relaxation": request.cognitive_state.relaxation,
            "valence": request.cognitive_state.valence,
        }
    }


# ============================================================================
# Text-to-Speech (ElevenLabs) Endpoint
# ============================================================================

class TTSRequest(BaseModel):
    """Request for text-to-speech conversion."""
    text: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default: Rachel voice


@app.post("/api/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using ElevenLabs API.

    Returns audio as MP3 bytes.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs API key not configured. Set ELEVENLABS_API_KEY environment variable."
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{request.voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": request.text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    }
                },
                timeout=30.0
            )

            if response.status_code != 200:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ElevenLabs API error: {response.text}"
                )

            # Return audio as MP3
            return Response(
                content=response.content,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=speech.mp3"}
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="ElevenLabs API timeout")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voices")
async def get_voices():
    """
    Get available ElevenLabs voices.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")

    if not api_key:
        # Return default voices without API
        return {
            "voices": [
                {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel (Default)"},
                {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
                {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
                {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},
                {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
                {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold"},
                {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
                {"voice_id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam"},
            ]
        }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key},
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "voices": [
                        {"voice_id": v["voice_id"], "name": v["name"]}
                        for v in data.get("voices", [])
                    ]
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch voices")

    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time EEG events.

    Events are broadcast to all clients by the background broadcast_events task.
    This endpoint just manages the connection lifecycle.

    Events sent to client:
    - connection: {status: "connected" | "disconnected"}
    - streaming: {status: "started" | "stopped"}
    - control_signal: {signal: "single_blink" | "double_blink" | ...}
    - attention: {focus: 0-1, relaxation: 0-1, engagement: 0-1}
    - eeg_raw: {channels: [[...], [...], ...], timestamp: float}
    """
    await ws_manager.connect(websocket)
    logger.info(f"WebSocket client connected (total: {len(ws_manager.active_connections)})")

    try:
        while True:
            try:
                # receive() handles all message types including disconnect gracefully
                await websocket.receive()
            except RuntimeError as e:
                # Handle "Cannot call receive once disconnect received"
                logger.debug(f"WebSocket receive error: {e}")
                break
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket)
        logger.info(f"WebSocket client disconnected (total: {len(ws_manager.active_connections)})")


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

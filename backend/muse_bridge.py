#!/usr/bin/env python3
"""Persistent Muse S Bluetooth bridge.

Runs as a standalone daemon that owns the BrainFlow BT connection.
Streams EEG data into shared memory so any backend process can read
without needing its own BT session. The Muse stays connected across
backend restarts.

Usage:
    python muse_bridge.py              # Real Muse S
    python muse_bridge.py --sim        # Simulated EEG

Architecture:
    muse_bridge.py  ──(BT)──►  Muse S headband
         │
         ├── SharedMemory ring buffer (4ch × 2560 samples = 10s @ 256Hz)
         ├── SharedMemory metadata (write_pos, total_samples, connected, timestamp)
         └── Unix socket /tmp/muse_bridge.sock (status queries, heartbeat)

    axiom_v2.py reads from shared memory via EEGBridgeClient (drop-in for EEGSource)
"""

import asyncio
import json
import os
import signal
import struct
import sys
import time

import numpy as np
from multiprocessing import shared_memory

SIM_MODE = "--sim" in sys.argv
EEG_SR = 256
N_CHANNELS = 4
RING_SECONDS = 10
RING_SAMPLES = EEG_SR * RING_SECONDS  # 2560

SHM_DATA_NAME = "muse_eeg_ring"
SHM_META_NAME = "muse_eeg_meta"
SOCKET_PATH = "/tmp/muse_bridge.sock"
PID_FILE = "/tmp/muse_bridge.pid"

# Metadata layout (64 bytes):
#   write_pos    (int64)   - current write position in ring
#   total        (int64)   - total samples written since start
#   connected    (int8)    - 1 if Muse connected, 0 if not
#   quality      (float32) - signal quality 0-1
#   timestamp    (float64) - last write time (time.time())
#   pid          (int32)   - bridge process PID
META_FORMAT = "<qqbfd i"
META_SIZE = struct.calcsize(META_FORMAT)  # Should be ~37 bytes


def cleanup_shm(name: str):
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


class MuseBridge:
    def __init__(self, sim=False):
        self.sim = sim
        self._board = None
        self._channels = None
        self._running = False
        self._total = 0

        # Clean up any stale shared memory
        cleanup_shm(SHM_DATA_NAME)
        cleanup_shm(SHM_META_NAME)

        # Create shared memory: ring buffer for EEG data
        ring_bytes = N_CHANNELS * RING_SAMPLES * 8  # float64
        self._shm_data = shared_memory.SharedMemory(
            name=SHM_DATA_NAME, create=True, size=ring_bytes
        )
        self._ring = np.ndarray(
            (N_CHANNELS, RING_SAMPLES), dtype=np.float64, buffer=self._shm_data.buf
        )
        self._ring[:] = 0

        # Create shared memory: metadata
        self._shm_meta = shared_memory.SharedMemory(
            name=SHM_META_NAME, create=True, size=max(META_SIZE, 64)
        )
        self._write_meta(0, 0, False, 0.0)

        print(f"[BRIDGE] Shared memory created: {SHM_DATA_NAME} ({ring_bytes} bytes)", flush=True)

    def _write_meta(self, write_pos: int, total: int, connected: bool, quality: float):
        data = struct.pack(META_FORMAT, write_pos, total, int(connected), quality, time.time(), os.getpid())
        self._shm_meta.buf[:len(data)] = data

    def _connect_muse(self):
        if self.sim:
            print("[BRIDGE] Simulation mode — no BT connection", flush=True)
            return True

        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

        params = BrainFlowInputParams()
        params.timeout = 15
        board_id = BoardIds.MUSE_S_BOARD.value

        try:
            self._board = BoardShim(board_id, params)
            print("[BRIDGE] Scanning for Muse S via Bluetooth...", flush=True)
            self._board.prepare_session()
            self._board.start_stream(num_samples=450000)
            self._channels = BoardShim.get_eeg_channels(board_id)[:4]
            print("[BRIDGE] Muse S connected, streaming 4ch @ 256Hz", flush=True)
            return True
        except Exception as e:
            print(f"[BRIDGE] Connection failed: {e}", flush=True)
            self._board = None
            return False

    def _disconnect_muse(self):
        if self._board:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass
            self._board = None
            print("[BRIDGE] Muse S disconnected", flush=True)

    def _pull_and_write(self):
        """Pull samples from BrainFlow/sim and write to shared memory ring."""
        if self.sim:
            n = int(EEG_SR * 0.05)  # ~13 samples per 50ms tick
            t = np.linspace(self._total / EEG_SR, (self._total + n) / EEG_SR, n)
            data = np.zeros((4, n))
            for ch in range(4):
                alpha = 15 * np.sin(2 * np.pi * 10 * t + ch)
                beta = 5 * np.sin(2 * np.pi * 20 * t + ch * 0.5)
                theta = 8 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
                noise = np.random.randn(n) * 3
                data[ch] = alpha + beta + theta + noise
        else:
            raw = self._board.get_board_data(num_samples=128)
            if raw.shape[1] == 0:
                return 0
            data = raw[self._channels, :]

        n = data.shape[1]
        pos = self._total % RING_SAMPLES

        # Write to ring buffer (handle wraparound)
        if pos + n <= RING_SAMPLES:
            self._ring[:, pos:pos + n] = data
        else:
            first = RING_SAMPLES - pos
            self._ring[:, pos:] = data[:, :first]
            self._ring[:, :n - first] = data[:, first:]

        self._total += n
        self._write_meta(self._total % RING_SAMPLES, self._total, True, 1.0)
        return n

    async def _stream_loop(self):
        """Main loop: pull EEG at 50ms intervals."""
        print("[BRIDGE] Streaming loop started", flush=True)
        while self._running:
            try:
                self._pull_and_write()
            except Exception as e:
                print(f"[BRIDGE] Stream error: {e}", flush=True)
                self._write_meta(self._total % RING_SAMPLES, self._total, False, 0.0)
                if not self.sim:
                    print("[BRIDGE] Attempting reconnect in 3s...", flush=True)
                    await asyncio.sleep(3)
                    if not self._connect_muse():
                        continue
                    self._write_meta(self._total % RING_SAMPLES, self._total, True, 1.0)
            await asyncio.sleep(0.05)

    async def _socket_server(self):
        """Unix socket for status queries and control commands."""
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        async def handle_client(reader, writer):
            try:
                data = await asyncio.wait_for(reader.read(1024), timeout=5)
                msg = data.decode().strip()

                if msg == "status":
                    resp = json.dumps({
                        "connected": self._board is not None or self.sim,
                        "sim": self.sim,
                        "total_samples": self._total,
                        "uptime_seconds": self._total / EEG_SR if self._total > 0 else 0,
                        "pid": os.getpid(),
                    })
                elif msg == "ping":
                    resp = "pong"
                elif msg == "stop":
                    resp = "stopping"
                    self._running = False
                else:
                    resp = json.dumps({"error": f"unknown command: {msg}"})

                writer.write(resp.encode())
                await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()

        server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
        print(f"[BRIDGE] Control socket at {SOCKET_PATH}", flush=True)

        while self._running:
            await asyncio.sleep(1)

        server.close()
        await server.wait_closed()

    def _write_pid(self):
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _cleanup(self):
        self._disconnect_muse()
        self._write_meta(0, self._total, False, 0.0)
        try:
            self._shm_data.close()
            self._shm_data.unlink()
        except Exception:
            pass
        try:
            self._shm_meta.close()
            self._shm_meta.unlink()
        except Exception:
            pass
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        if os.path.exists(PID_FILE):
            os.unlink(PID_FILE)
        print("[BRIDGE] Cleaned up", flush=True)

    async def run(self):
        self._running = True
        self._write_pid()

        if not self._connect_muse():
            if not self.sim:
                print("[BRIDGE] Could not connect to Muse S. Exiting.", flush=True)
                self._cleanup()
                return

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: setattr(self, '_running', False))

        try:
            await asyncio.gather(
                self._stream_loop(),
                self._socket_server(),
            )
        finally:
            self._cleanup()


class EEGBridgeClient:
    """Drop-in replacement for EEGSource that reads from the bridge's shared memory.

    Same interface: start(), stop(), pull(), get_window(seconds).
    """

    def __init__(self, sim=False):
        self._sim = sim
        self._shm_data = None
        self._shm_meta = None
        self._ring = None
        self._local_ring = np.zeros((N_CHANNELS, EEG_SR * RING_SECONDS))
        self._local_pos = 0
        self._local_total = 0
        self._last_read_total = 0
        self._connected = False

    def _read_meta(self) -> dict:
        if self._shm_meta is None:
            return {"write_pos": 0, "total": 0, "connected": False, "quality": 0.0, "timestamp": 0.0, "pid": 0}
        data = bytes(self._shm_meta.buf[:META_SIZE])
        wp, total, connected, quality, ts, pid = struct.unpack(META_FORMAT, data)
        return {
            "write_pos": wp,
            "total": total,
            "connected": bool(connected),
            "quality": quality,
            "timestamp": ts,
            "pid": pid,
        }

    def start(self):
        if self._sim:
            print("[EEG] Simulation mode (no bridge needed)", flush=True)
            return

        try:
            self._shm_data = shared_memory.SharedMemory(name=SHM_DATA_NAME, create=False)
            self._shm_meta = shared_memory.SharedMemory(name=SHM_META_NAME, create=False)
            self._ring = np.ndarray(
                (N_CHANNELS, RING_SAMPLES), dtype=np.float64, buffer=self._shm_data.buf
            )
            meta = self._read_meta()
            self._last_read_total = meta["total"]
            self._connected = True
            print(f"[EEG] Connected to muse_bridge (pid={meta['pid']}, total={meta['total']})", flush=True)
        except FileNotFoundError:
            print("[EEG] ERROR: muse_bridge not running. Start it first: python muse_bridge.py", flush=True)
            print("[EEG] Falling back to simulation mode", flush=True)
            self._sim = True

    def stop(self):
        if self._shm_data:
            self._shm_data.close()
            self._shm_data = None
        if self._shm_meta:
            self._shm_meta.close()
            self._shm_meta = None
        self._ring = None

    def pull(self) -> np.ndarray:
        if self._sim:
            n = int(EEG_SR * 0.05)
            t = np.linspace(self._local_total / EEG_SR, (self._local_total + n) / EEG_SR, n)
            data = np.zeros((4, n))
            for ch in range(4):
                alpha = 15 * np.sin(2 * np.pi * 10 * t + ch)
                beta = 5 * np.sin(2 * np.pi * 20 * t + ch * 0.5)
                theta = 8 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
                noise = np.random.randn(n) * 3
                data[ch] = alpha + beta + theta + noise
            self._local_total += n
            self._write_local_ring(data)
            return data

        meta = self._read_meta()
        bridge_total = meta["total"]
        new_samples = bridge_total - self._last_read_total

        if new_samples <= 0:
            return np.zeros((4, 0))

        # Cap to ring size to avoid reading stale wrapped-around data
        new_samples = min(new_samples, RING_SAMPLES)
        end_pos = meta["write_pos"]
        start_pos = (end_pos - new_samples) % RING_SAMPLES

        if start_pos < end_pos:
            data = self._ring[:, start_pos:end_pos].copy()
        else:
            data = np.concatenate([
                self._ring[:, start_pos:],
                self._ring[:, :end_pos]
            ], axis=1)

        self._last_read_total = bridge_total
        self._write_local_ring(data)
        return data

    def _write_local_ring(self, data):
        n = data.shape[1]
        cap = self._local_ring.shape[1]
        for i in range(n):
            self._local_ring[:, self._local_pos % cap] = data[:, i]
            self._local_pos += 1
        self._local_total = max(self._local_total, self._local_pos)

    def get_window(self, seconds: float) -> np.ndarray:
        n = int(seconds * EEG_SR)
        cap = self._local_ring.shape[1]
        n = min(n, self._local_pos, cap)
        if n <= 0:
            return np.zeros((4, 0))
        end = self._local_pos % cap
        if n <= end:
            return self._local_ring[:, end - n:end].copy()
        return np.concatenate([
            self._local_ring[:, cap - (n - end):],
            self._local_ring[:, :end]
        ], axis=1)

    @property
    def bridge_status(self) -> dict:
        return self._read_meta()


# ── CLI helpers ───────────────────────────────────────────────

def is_bridge_running() -> bool:
    """Check if a bridge process is already running."""
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Check if process exists
        return True
    except (ProcessLookupError, ValueError):
        os.unlink(PID_FILE)
        return False


def bridge_status() -> dict | None:
    """Query bridge status via Unix socket."""
    import socket
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(SOCKET_PATH)
        sock.sendall(b"status")
        data = sock.recv(4096)
        sock.close()
        return json.loads(data.decode())
    except Exception:
        return None


if __name__ == "__main__":
    if "--status" in sys.argv:
        status = bridge_status()
        if status:
            print(f"Bridge running (pid={status['pid']})")
            print(f"  Connected: {status['connected']}")
            print(f"  Sim: {status['sim']}")
            print(f"  Total samples: {status['total_samples']}")
            print(f"  Uptime: {status['uptime_seconds']:.1f}s")
        else:
            print("Bridge not running")
        sys.exit(0)

    if "--stop" in sys.argv:
        status = bridge_status()
        if status:
            import socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(SOCKET_PATH)
            sock.sendall(b"stop")
            print(f"Stopping bridge (pid={status['pid']})")
            sock.close()
        else:
            print("Bridge not running")
        sys.exit(0)

    if is_bridge_running():
        print("[BRIDGE] Already running! Use --status to check or --stop to kill it.", flush=True)
        sys.exit(1)

    print("═══════════════════════════════════════════", flush=True)
    print("  MUSE BRIDGE — Persistent BT Connection", flush=True)
    print("═══════════════════════════════════════════", flush=True)
    print(f"  Mode: {'Simulation' if SIM_MODE else 'Real Muse S'}", flush=True)
    print(f"  Ring: {N_CHANNELS}ch × {RING_SAMPLES} samples ({RING_SECONDS}s)", flush=True)
    print(f"  Socket: {SOCKET_PATH}", flush=True)
    print("", flush=True)

    bridge = MuseBridge(sim=SIM_MODE)
    asyncio.run(bridge.run())

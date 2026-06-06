#!/usr/bin/env python3
"""Persistent Muse S Bluetooth bridge.

Uses bleak (Python-native BLE) for the Bluetooth connection instead of
BrainFlow's SimpleBLE, which hangs on macOS GATT handshakes.

Streams EEG data into shared memory so any backend process can read
without needing its own BT session. The Muse stays connected across
backend restarts.

Usage:
    python muse_bridge.py              # Real Muse S
    python muse_bridge.py --sim        # Simulated EEG

Architecture:
    muse_bridge.py  ──(bleak BLE)──►  Muse S headband
         │
         ├── SharedMemory ring buffer (4ch × 2560 samples = 10s @ 256Hz)
         ├── SharedMemory metadata (write_pos, total_samples, connected, timestamp)
         └── Unix socket /tmp/muse_bridge.sock (status queries, heartbeat)

    axiom_v2.py reads from shared memory via EEGBridgeClient (drop-in for EEGSource)
"""

import asyncio
import atexit
import json
import os
import signal
import struct
import sys
import time

import numpy as np
from multiprocessing import shared_memory
from bleak import BleakScanner, BleakClient

SIM_MODE = "--sim" in sys.argv
EEG_SR = 256
N_CHANNELS = 4
RING_SECONDS = 10
RING_SAMPLES = EEG_SR * RING_SECONDS

SHM_DATA_NAME = "muse_eeg_ring"
SHM_META_NAME = "muse_eeg_meta"
SOCKET_PATH = "/tmp/muse_bridge.sock"
PID_FILE = "/tmp/muse_bridge.pid"

BLE_SCAN_TIMEOUT = 10
BLE_MAX_RETRIES = 3

MUSE_SERVICE = '0000fe8d-0000-1000-8000-00805f9b34fb'
CONTROL_CHAR = '273e0001-4c4d-454d-96be-f03bac821358'
EEG_CHARS = [
    '273e0003-4c4d-454d-96be-f03bac821358',  # TP9
    '273e0004-4c4d-454d-96be-f03bac821358',  # AF7
    '273e0005-4c4d-454d-96be-f03bac821358',  # AF8
    '273e0006-4c4d-454d-96be-f03bac821358',  # TP10
]

META_FORMAT = "<qqbfd i"
META_SIZE = struct.calcsize(META_FORMAT)


def cleanup_shm(name: str):
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


def _decode_eeg_packet(data: bytes) -> list[float]:
    """Decode a Muse S EEG BLE packet: 2-byte timestamp + 12 × 12-bit samples."""
    if len(data) < 20:
        return []
    bits = int.from_bytes(data[2:], 'big')
    samples = []
    for i in range(12):
        shift = (11 - i) * 12
        raw = (bits >> shift) & 0xFFF
        uv = (raw - 2048) * 0.48828125
        samples.append(uv)
    return samples


class MuseBridge:
    def __init__(self, sim=False):
        self.sim = sim
        self._client: BleakClient | None = None
        self._running = False
        self._total = 0
        self._muse_address: str | None = None

        cleanup_shm(SHM_DATA_NAME)
        cleanup_shm(SHM_META_NAME)

        ring_bytes = N_CHANNELS * RING_SAMPLES * 8
        self._shm_data = shared_memory.SharedMemory(
            name=SHM_DATA_NAME, create=True, size=ring_bytes
        )
        self._ring = np.ndarray(
            (N_CHANNELS, RING_SAMPLES), dtype=np.float64, buffer=self._shm_data.buf
        )
        self._ring[:] = 0

        self._shm_meta = shared_memory.SharedMemory(
            name=SHM_META_NAME, create=True, size=max(META_SIZE, 64)
        )
        self._write_meta(0, 0, False, 0.0)

        print(f"[BRIDGE] Shared memory created: {SHM_DATA_NAME} ({ring_bytes} bytes)", flush=True)

        atexit.register(self._atexit_cleanup)

    def _write_meta(self, write_pos: int, total: int, connected: bool, quality: float):
        data = struct.pack(META_FORMAT, write_pos, total, int(connected), quality, time.time(), os.getpid())
        self._shm_meta.buf[:len(data)] = data

    def _write_samples(self, ch_idx: int, samples: list[float]):
        """Write decoded EEG samples for one channel into the shared ring buffer."""
        for uv in samples:
            pos = self._total % RING_SAMPLES
            self._ring[ch_idx, pos] = uv
        # Note: _total is incremented per-packet in the notification handler
        # after all 4 channels are written

    async def _find_muse(self) -> str | None:
        """Scan for a Muse device and return its BLE address."""
        print(f"[BRIDGE] Scanning for Muse S ({BLE_SCAN_TIMEOUT}s)...", flush=True)
        devices = await BleakScanner.discover(timeout=BLE_SCAN_TIMEOUT, return_adv=True)

        for addr, (dev, adv) in devices.items():
            name = dev.name or ''
            svc_uuids = [str(u).lower() for u in (adv.service_uuids or [])]
            if 'muse' in name.lower() or MUSE_SERVICE in svc_uuids:
                print(f"[BRIDGE] Found {name} @ {dev.address}", flush=True)
                return dev.address

        print(f"[BRIDGE] No Muse found among {len(devices)} BLE devices", flush=True)
        return None

    async def _connect_muse(self) -> bool:
        if self.sim:
            print("[BRIDGE] Simulation mode — no BT connection", flush=True)
            return True

        for attempt in range(1, BLE_MAX_RETRIES + 1):
            if not self._running:
                return False

            print(f"[BRIDGE] Connection attempt {attempt}/{BLE_MAX_RETRIES}...", flush=True)

            address = await self._find_muse()
            if not address:
                if attempt < BLE_MAX_RETRIES:
                    print("[BRIDGE] Retrying in 3s...", flush=True)
                    await asyncio.sleep(3)
                continue

            try:
                def on_disconnect(c):
                    print("[BRIDGE] BLE disconnected callback fired", flush=True)

                client = BleakClient(address, timeout=15, disconnected_callback=on_disconnect)
                await client.connect()
                print("[BRIDGE] BLE connected", flush=True)

                # Start EEG streaming: send 'd' (start) then 'p21' (EEG preset)
                await client.write_gatt_char(CONTROL_CHAR, b'\x02\x64\x0a', response=False)
                await asyncio.sleep(0.1)
                await client.write_gatt_char(CONTROL_CHAR, b'\x04\x70\x32\x31\x0a', response=False)
                await asyncio.sleep(0.1)

                self._ch_write_pos = [0, 0, 0, 0]

                def make_handler(ch_idx):
                    def handler(sender, data):
                        samples = _decode_eeg_packet(bytes(data))
                        if not samples:
                            return
                        pos = self._ch_write_pos[ch_idx]
                        for s in samples:
                            self._ring[ch_idx, pos % RING_SAMPLES] = s
                            pos += 1
                        self._ch_write_pos[ch_idx] = pos
                        # Advance shared write pointer to the min across all channels
                        min_pos = min(self._ch_write_pos)
                        if min_pos > self._total:
                            self._total = min_pos
                            self._write_meta(self._total % RING_SAMPLES, self._total, True, 1.0)
                    return handler

                for i, char_uuid in enumerate(EEG_CHARS):
                    await client.start_notify(char_uuid, make_handler(i))

                self._client = client
                self._muse_address = address
                print("[BRIDGE] Muse S connected, streaming 4ch EEG via bleak", flush=True)
                return True

            except Exception as e:
                print(f"[BRIDGE] Connection failed: {e}", flush=True)
                try:
                    await client.disconnect()
                except Exception:
                    pass
                if attempt < BLE_MAX_RETRIES:
                    print("[BRIDGE] Retrying in 3s...", flush=True)
                    await asyncio.sleep(3)

        print(f"[BRIDGE] All {BLE_MAX_RETRIES} attempts failed", flush=True)
        return False

    async def _disconnect_muse(self):
        if self._client:
            try:
                # Send halt command
                await self._client.write_gatt_char(CONTROL_CHAR, b'\x02\x68\x0a', response=False)
                await asyncio.sleep(0.1)
            except Exception:
                pass
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
            print("[BRIDGE] Muse S disconnected", flush=True)

    def _atexit_cleanup(self):
        if self._client and self._client.is_connected:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._client.disconnect())
                else:
                    loop.run_until_complete(self._client.disconnect())
            except Exception:
                pass

    async def _stream_loop(self):
        """Monitor connection health. Bleak streams via BLE notifications
        (handled in _connect_muse callbacks), so this loop just checks
        that the connection is alive and handles sim mode."""
        print("[BRIDGE] Streaming loop started", flush=True)
        last_total = 0
        stall_count = 0

        while self._running:
            if self.sim:
                n = int(EEG_SR * 0.05)
                t = np.linspace(self._total / EEG_SR, (self._total + n) / EEG_SR, n)
                for ch in range(4):
                    data = (15 * np.sin(2 * np.pi * 10 * t + ch)
                            + 5 * np.sin(2 * np.pi * 20 * t + ch * 0.5)
                            + 8 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
                            + np.random.randn(n) * 3)
                    pos = self._total % RING_SAMPLES
                    end = pos + n
                    if end <= RING_SAMPLES:
                        self._ring[ch, pos:end] = data
                    else:
                        first = RING_SAMPLES - pos
                        self._ring[ch, pos:] = data[:first]
                        self._ring[ch, :n - first] = data[first:]
                self._total += n
                self._write_meta(self._total % RING_SAMPLES, self._total, True, 1.0)
                await asyncio.sleep(0.05)
                continue

            # Real mode: check if BLE notifications are still arriving
            if self._client and not self._client.is_connected:
                print("[BRIDGE] BLE connection lost", flush=True)
                self._write_meta(self._total % RING_SAMPLES, self._total, False, 0.0)
                await self._disconnect_muse()
                print("[BRIDGE] Attempting reconnect...", flush=True)
                if not await self._connect_muse():
                    print("[BRIDGE] Reconnect failed, exiting.", flush=True)
                    self._running = False
                    return

            # Send keepalive every ~10s to prevent Muse from sleeping
            if self._client and self._client.is_connected and stall_count % 100 == 50:
                try:
                    await self._client.write_gatt_char(CONTROL_CHAR, b'\x02\x6b\x0a', response=False)
                except Exception:
                    pass

            # Check for data stall (no new samples for 10s)
            if self._total == last_total:
                stall_count += 1
                if stall_count >= 100:  # 100 × 0.1s = 10s
                    print("[BRIDGE] Data stall detected (no samples for 10s)", flush=True)
                    self._write_meta(self._total % RING_SAMPLES, self._total, False, 0.0)
                    await self._disconnect_muse()
                    print("[BRIDGE] Attempting reconnect...", flush=True)
                    if not await self._connect_muse():
                        print("[BRIDGE] Reconnect failed, exiting.", flush=True)
                        self._running = False
                        return
                    stall_count = 0
            else:
                stall_count = 0
                last_total = self._total

            await asyncio.sleep(0.1)

    async def _socket_server(self):
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        async def handle_client(reader, writer):
            try:
                data = await asyncio.wait_for(reader.read(1024), timeout=5)
                msg = data.decode().strip()

                if msg == "status":
                    resp = json.dumps({
                        "connected": (self._client is not None and self._client.is_connected) or self.sim,
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

    async def _cleanup(self):
        await self._disconnect_muse()
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

    def _signal_handler(self):
        print("[BRIDGE] Shutdown signal received", flush=True)
        self._running = False

    async def run(self):
        self._running = True
        self._write_pid()

        if not await self._connect_muse():
            if not self.sim:
                print("[BRIDGE] Could not connect to Muse S. Exiting.", flush=True)
                await self._cleanup()
                return

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)

        try:
            await asyncio.gather(
                self._stream_loop(),
                self._socket_server(),
            )
        finally:
            await self._cleanup()


class EEGBridgeClient:
    """Drop-in replacement for EEGSource that reads from the bridge's shared memory."""

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
        return {"write_pos": wp, "total": total, "connected": bool(connected),
                "quality": quality, "timestamp": ts, "pid": pid}

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
                data[ch] = (15 * np.sin(2 * np.pi * 10 * t + ch)
                            + 5 * np.sin(2 * np.pi * 20 * t + ch * 0.5)
                            + 8 * np.sin(2 * np.pi * 6 * t + ch * 0.3)
                            + np.random.randn(n) * 3)
            self._local_total += n
            self._write_local_ring(data)
            return data

        meta = self._read_meta()
        new_samples = meta["total"] - self._last_read_total
        if new_samples <= 0:
            return np.zeros((4, 0))

        new_samples = min(new_samples, RING_SAMPLES)
        end_pos = meta["write_pos"]
        start_pos = (end_pos - new_samples) % RING_SAMPLES

        if start_pos < end_pos:
            data = self._ring[:, start_pos:end_pos].copy()
        else:
            data = np.concatenate([self._ring[:, start_pos:], self._ring[:, :end_pos]], axis=1)

        self._last_read_total = meta["total"]
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
        n = min(int(seconds * EEG_SR), self._local_pos, self._local_ring.shape[1])
        if n <= 0:
            return np.zeros((4, 0))
        cap = self._local_ring.shape[1]
        end = self._local_pos % cap
        if n <= end:
            return self._local_ring[:, end - n:end].copy()
        return np.concatenate([self._local_ring[:, cap - (n - end):], self._local_ring[:, :end]], axis=1)

    @property
    def bridge_status(self) -> dict:
        return self._read_meta()


# ── CLI helpers ───────────────────────────────────────────────

def is_bridge_running() -> bool:
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        try:
            os.unlink(PID_FILE)
        except OSError:
            pass
        return False


def bridge_status() -> dict | None:
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
            import socket as sock_mod
            sock = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
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
    print("  (bleak BLE — no BrainFlow for BT)", flush=True)
    print("═══════════════════════════════════════════", flush=True)
    print(f"  Mode: {'Simulation' if SIM_MODE else 'Real Muse S'}", flush=True)
    print(f"  Ring: {N_CHANNELS}ch × {RING_SAMPLES} samples ({RING_SECONDS}s)", flush=True)
    print("", flush=True)

    bridge = MuseBridge(sim=SIM_MODE)
    asyncio.run(bridge.run())

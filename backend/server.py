#!/usr/bin/env python3
"""WebSocket server: streams Muse S EEG + RL state vectors.

Two message types sent to clients:
  - "eeg"   (20Hz): filtered waveforms + band powers for display
  - "state" (1Hz):  29-dim RL state vector with features, ratios, coherence
"""

import asyncio
import json
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations,
    NoiseTypes, WindowOperations,
)

import websockets
from features import extract_state, BANDS

BOARD_ID = BoardIds.MUSE_S_BOARD.value
EEG_SR = 256
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

board = None
clients: set = set()

eeg_ring = np.zeros((4, EEG_SR * 10))
ring_pos = 0
total_samples = 0
prev_bands = None


def connect_muse():
    global board
    params = BrainFlowInputParams()
    params.timeout = 5
    BoardShim.enable_board_logger()
    board = BoardShim(BOARD_ID, params)
    print("Scanning for Muse S...", flush=True)
    board.prepare_session()
    print("Connected!", flush=True)
    board.start_stream(num_samples=450000)
    print("Streaming EEG at 256 Hz", flush=True)


def filter_channel(data, sr=EEG_SR):
    out = data.copy()
    if len(out) < 12:
        return out
    DataFilter.detrend(out, DetrendOperations.LINEAR.value)
    DataFilter.perform_bandpass(out, sr, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0.0)
    DataFilter.remove_environmental_noise(out, sr, NoiseTypes.SIXTY.value)
    return out


def get_ring_slice(n_samples):
    cap = eeg_ring.shape[1]
    n = min(n_samples, total_samples, cap)
    if n <= 0:
        return np.zeros((4, 0))
    end = ring_pos % cap
    if n <= end:
        return eeg_ring[:, end - n:end].copy()
    return np.concatenate([eeg_ring[:, cap - (n - end):], eeg_ring[:, :end]], axis=1)


def compute_bands_display(filtered_4ch):
    band_list = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0), ("gamma", 30.0, 50.0)]
    result = {name: 0.0 for name, _, _ in band_list}
    count = 0
    for ch in range(4):
        d = filtered_4ch[ch]
        if len(d) < EEG_SR:
            continue
        nfft = DataFilter.get_nearest_power_of_two(EEG_SR)
        psd = DataFilter.get_psd_welch(d, nfft, nfft // 2, EEG_SR, WindowOperations.HANNING.value)
        for name, lo, hi in band_list:
            result[name] += float(DataFilter.get_band_power(psd, lo, hi))
        count += 1
    if count > 0:
        for name in result:
            result[name] /= count
    return result


async def stream_loop():
    global ring_pos, total_samples, prev_bands

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    state_tick = 0

    while True:
        raw = board.get_board_data(num_samples=128)
        if raw.shape[1] == 0:
            await asyncio.sleep(0.02)
            continue

        n = raw.shape[1]
        eeg = raw[eeg_channels[:4], :]

        cap = eeg_ring.shape[1]
        for i in range(n):
            eeg_ring[:, ring_pos % cap] = eeg[:, i]
            ring_pos += 1
        total_samples += n

        # Filtered display data (last 2s)
        segment = get_ring_slice(EEG_SR * 2)
        filtered = np.zeros_like(segment)
        for ch in range(4):
            filtered[ch] = filter_channel(segment[ch])

        bands = compute_bands_display(filtered)

        step = 4
        eeg_msg = json.dumps({
            "type": "eeg",
            "channels": CHANNELS,
            "data": filtered[:, ::step].tolist(),
            "bands": bands,
            "sampleRate": EEG_SR // step,
            "totalSamples": total_samples,
            "timestamp": time.time(),
        })

        # RL state vector at 1Hz
        state_msg = None
        state_tick += n
        if state_tick >= EEG_SR:
            state_tick = 0
            state_seg = get_ring_slice(EEG_SR * 2)
            result = extract_state(state_seg, prev_bands)
            prev_bands = result["bands"]
            state_msg = json.dumps({
                "type": "state",
                **result,
                "timestamp": time.time(),
                "totalSamples": total_samples,
            })

        dead = set()
        for ws in list(clients):
            try:
                await ws.send(eeg_msg)
                if state_msg:
                    await ws.send(state_msg)
            except Exception:
                dead.add(ws)
        for d in dead:
            clients.discard(d)

        await asyncio.sleep(0.05)


async def handler(ws):
    clients.add(ws)
    print(f"Client connected ({len(clients)} total)", flush=True)
    try:
        await ws.send(json.dumps({
            "type": "info",
            "channels": CHANNELS,
            "sampleRate": EEG_SR,
            "bands": [name for name, _, _ in BANDS],
            "stateSize": 29,
        }))
        async for _ in ws:
            pass
    finally:
        clients.discard(ws)
        print(f"Client disconnected ({len(clients)} total)", flush=True)


async def main():
    connect_muse()
    server = await websockets.serve(handler, "127.0.0.1", 8080)
    print("WebSocket server on ws://127.0.0.1:8080", flush=True)
    asyncio.create_task(stream_loop())
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())

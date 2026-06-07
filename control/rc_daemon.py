#!/usr/bin/env python3
"""
rc_daemon.py — robot car control daemon
Single shared BLE + USB connection, WebSocket API on ws://localhost:8766
"""

import asyncio, atexit, json, logging, os, signal, subprocess, sys, time
import serial, serial.tools.list_ports
from bleak import BleakClient, BleakScanner
import websockets

BLE_NAMES      = {"SH-HC-08", "YOGI-BLE1", "HMSoft", "HM-10", "HMSoft_BLE"}
BLE_SVC        = "0000ffe0-0000-1000-8000-00805f9b34fb"
BLE_CHAR       = "0000ffe1-0000-1000-8000-00805f9b34fb"
BLE_MAC        = "C8:DF:84:2A:3A:B2"
WS_PORT        = 8766
BAUD           = 9600
SCAN_TIMEOUT   = 8.0
USB_KEYS       = ("usbmodem", "usbserial", "arduino")
SKIP_PORTS     = ("bluetooth-incoming-port", "debug-console")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("/tmp/rc_daemon.log")]
)
log = logging.getLogger("rcd")


def find_usb():
    for p in serial.tools.list_ports.comports():
        d = p.device
        if not d.startswith("/dev/cu."): continue
        if any(s in d.lower() for s in SKIP_PORTS): continue
        if any(k in d.lower() or k in (p.description or "").lower() for k in USB_KEYS):
            return d
    return None


def clear_stale_ble():
    try:
        r = subprocess.run(["blueutil", "--is-connected", BLE_MAC],
                           capture_output=True, text=True, timeout=5)
        if r.stdout.strip() == "1":
            log.info("Clearing stale BLE connection via blueutil")
            subprocess.run(["blueutil", "--power", "off"], timeout=5)
            time.sleep(2)
            subprocess.run(["blueutil", "--power", "on"], timeout=5)
            time.sleep(3)
            return True
    except Exception as e:
        log.warning("blueutil error: %s", e)
    return False


class RCDaemon:
    def __init__(self):
        self.ble_client  = None
        self.ble_device  = None
        self.ser         = None
        self.ws_clients  = set()
        self._closing    = False

    @property
    def ble_ok(self):
        return self.ble_client and self.ble_client.is_connected

    @property
    def usb_ok(self):
        if not self.ser:
            return False
        try:
            return self.ser.is_open and self.ser.in_waiting >= 0
        except Exception:
            self._close_usb()
            return False

    def _close_usb(self):
        if self.ser:
            try: self.ser.close()
            except: pass
            self.ser = None

    @property
    def mode(self):
        if self.usb_ok and self.ble_ok: return "both"
        if self.ble_ok: return "ble"
        if self.usb_ok: return "usb"
        return "none"

    # ── broadcast ────────────────────────────────────────────────────
    async def broadcast(self, obj):
        dead = set()
        msg = json.dumps(obj)
        for ws in list(self.ws_clients):
            try:    await ws.send(msg)
            except: dead.add(ws)
        self.ws_clients -= dead

    async def push_status(self, msg):
        m = self.mode
        log.info("[%s] %s", m, msg)
        await self.broadcast({"type": "status", "mode": m,
                              "device": self.ble_device.name if self.ble_device else None,
                              "msg": msg})

    # ── send command (send to ALL available channels) ──────────────
    async def send(self, text):
        line = (text.strip() + "\n").encode()
        sent = False
        if self.usb_ok:
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: (self.ser.write(line), self.ser.flush())),
                    timeout=2.0)
                sent = True
            except Exception as e:
                log.warning("USB write fail: %s — clearing", e)
                self._close_usb()
        if self.ble_ok:
            try:
                await self.ble_client.write_gatt_char(BLE_CHAR, line, response=False)
                sent = True
            except Exception as e:
                log.warning("BLE write fail: %s", e)
        if not sent:
            log.warning("No channel available for: %s", text.strip())

    # ── BLE ──────────────────────────────────────────────────────────
    def _on_ble_data(self, _sender, data):
        text = data.decode("ascii", errors="ignore").strip()
        if text:
            log.info("BLE rx: %s", text)
            asyncio.create_task(
                self.broadcast({"type": "data", "text": text}))

    def _on_ble_disconnect(self, client):
        if self._closing: return
        log.warning("BLE dropped")
        self.ble_client = None
        asyncio.create_task(self.push_status("BLE lost — reconnecting"))

    async def ble_connect(self):
        found = None
        def cb(device, adv):
            nonlocal found
            if found: return
            if device.name and (device.name in BLE_NAMES or "hm" in device.name.lower()):
                found = device
            elif adv.service_uuids and BLE_SVC in [u.lower() for u in adv.service_uuids]:
                found = device

        async with BleakScanner(cb):
            deadline = asyncio.get_event_loop().time() + SCAN_TIMEOUT
            while not found and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.2)

        if not found:
            return False

        log.info("BLE found: %s @ %s", found.name, found.address)
        try:
            client = BleakClient(found,
                                 disconnected_callback=self._on_ble_disconnect,
                                 timeout=10.0)
            await client.connect()
            await client.start_notify(BLE_CHAR, self._on_ble_data)
            self.ble_client = client
            self.ble_device = found
            log.info("BLE connected to %s", found.name)

            # Verify with a ping
            await client.write_gatt_char(BLE_CHAR, b"?\n", response=False)
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            log.warning("BLE connect failed: %s", e)
            try: await client.disconnect()
            except: pass
            return False

    async def ble_keepalive(self):
        while not self._closing:
            await asyncio.sleep(4)
            if self.ble_ok:
                try:
                    await self.ble_client.write_gatt_char(
                        BLE_CHAR, b"?\n", response=False)
                except Exception as e:
                    log.warning("BLE keepalive fail: %s", e)

    async def ble_loop(self):
        backoff = 2.0
        stale_cleared = False
        while not self._closing:
            if self.ble_ok:
                await asyncio.sleep(2)
                continue

            self.ble_client = None
            await self.push_status("BLE scanning...")

            ok = await self.ble_connect()
            if ok:
                backoff = 2.0
                stale_cleared = False
                await self.push_status(f"BLE connected to {self.ble_device.name}")
            else:
                if not stale_cleared:
                    log.info("BLE not found — clearing stale connections")
                    await asyncio.get_event_loop().run_in_executor(None, clear_stale_ble)
                    stale_cleared = True
                    await asyncio.sleep(2)
                    continue
                await self.push_status(f"BLE not found, retry in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 20.0)

    # ── USB ──────────────────────────────────────────────────────────
    async def _usb_read_loop(self):
        buf = b""
        while self.ser and not self._closing:
            try:
                n = self.ser.in_waiting
                if n:
                    chunk = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.ser.read(n))
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        text = line.decode("ascii", errors="ignore").strip()
                        if text:
                            await self.broadcast({"type": "data", "text": text})
                else:
                    await asyncio.sleep(0.03)
            except Exception:
                log.warning("USB read loop error — clearing")
                self._close_usb()
                await self.push_status("USB disconnected")
                break

    async def usb_loop(self):
        while not self._closing:
            port = find_usb()

            if port and not self.usb_ok:
                try:
                    ser = serial.Serial()
                    ser.port = port
                    ser.baudrate = BAUD
                    ser.timeout = 1
                    ser.dtr = False
                    ser.open()
                    await asyncio.sleep(0.3)
                    ser.reset_input_buffer()
                    ser.write(b"S\n"); ser.flush()
                    await asyncio.sleep(0.2)
                    ser.write(b"?\n"); ser.flush()
                    await asyncio.sleep(0.5)
                    resp = ser.readline().decode("ascii", errors="ignore").strip()
                    if "READY" in resp or "STOP" in resp:
                        self.ser = ser
                        log.info("USB connected on %s", port)
                        asyncio.create_task(self._usb_read_loop())
                        await self.push_status(f"USB connected on {port}")
                    else:
                        ser.close()
                except Exception as e:
                    log.warning("USB error: %s", e)

            elif not port and self.ser:
                log.info("USB disconnected")
                self._close_usb()
                await self.push_status("USB disconnected")

            await asyncio.sleep(3)

    # ── WebSocket ────────────────────────────────────────────────────
    async def _handle_ws(self, ws):
        self.ws_clients.add(ws)
        log.info("WS +1 (%d total)", len(self.ws_clients))
        await ws.send(json.dumps({
            "type": "status", "mode": self.mode,
            "device": self.ble_device.name if self.ble_device else None,
            "msg": f"connected, mode={self.mode}"
        }))
        try:
            async for raw in ws:
                cmd = raw.strip()
                if cmd:
                    await self.send(cmd)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.ws_clients.discard(ws)
            log.info("WS -1 (%d remain)", len(self.ws_clients))
            if len(self.ws_clients) == 0:
                await self.send("S")
                log.info("All clients gone — STOP")

    # ── clean shutdown ───────────────────────────────────────────────
    async def shutdown(self):
        if self._closing: return
        self._closing = True
        log.info("Shutting down...")
        await self.send("S")
        if self.ble_client:
            try:
                await self.ble_client.disconnect()
                log.info("BLE disconnected cleanly")
            except: pass
            self.ble_client = None
        if self.ser:
            try: self.ser.close()
            except: pass
            self.ser = None

    # ── run ──────────────────────────────────────────────────────────
    async def run(self):
        log.info("rc_daemon starting — ws://localhost:%d", WS_PORT)

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        async with websockets.serve(self._handle_ws, "localhost", WS_PORT):
            await asyncio.gather(
                self.ble_loop(), self.ble_keepalive(), self.usb_loop())


if __name__ == "__main__":
    daemon = RCDaemon()
    atexit.register(lambda: subprocess.run(
        ["blueutil", "--disconnect", BLE_MAC],
        capture_output=True, timeout=5) if BLE_MAC else None)
    try:
        asyncio.run(daemon.run())
    except (KeyboardInterrupt, SystemExit):
        pass
    log.info("rc_daemon stopped")

#!/usr/bin/env python3
import base64
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

BRIDGE_HOST = "127.0.0.1"
BRIDGE_PORT = int(os.getenv("PCC_BRIDGE_PORT", "8765"))
OUT_DIR = Path(os.getenv("OUT_DIR", str(Path.cwd() / "runs" / "screenshots")))
COUNT = int(os.getenv("COUNT", "5"))
INTERVAL = float(os.getenv("INTERVAL", "1.5"))
TIMEOUT = float(os.getenv("TIMEOUT", "40"))  # seconds to wait for viewer

OUT_DIR.mkdir(parents=True, exist_ok=True)


def send_lenpref_json(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_lenpref_json(sock: socket.socket) -> dict:
    hdr = sock.recv(4)
    if not hdr or len(hdr) < 4:
        return {}
    (length,) = struct.unpack("!I", hdr)
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            break
        payload += chunk
    if not payload:
        return {}
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return {}


def wait_for_bridge(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((BRIDGE_HOST, port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> int:
    port = BRIDGE_PORT
    if not wait_for_bridge(port, TIMEOUT):
        print(f"❌ Could not connect to viewer bridge on {BRIDGE_HOST}:{port} within {TIMEOUT}s", file=sys.stderr)
        return 2

    saved = 0
    with socket.create_connection((BRIDGE_HOST, port), timeout=5) as s:
        for i in range(COUNT):
            # Request screenshot
            send_lenpref_json(s, {"type": "screenshot_request"})
            reply = recv_lenpref_json(s)
            b64 = reply.get("png_base64", "")
            ts = int(time.time() * 1000)
            out_path = OUT_DIR / f"miniplanet_{os.getpid()}_{ts}_{i+1}.png"
            if b64:
                try:
                    png_bytes = base64.b64decode(b64)
                    with open(out_path, "wb") as f:
                        f.write(png_bytes)
                    print(f"✅ Saved screenshot: {out_path}")
                    saved += 1
                except Exception as e:
                    print(f"⚠️ Failed to save screenshot {i+1}: {e}", file=sys.stderr)
            else:
                print(f"⚠️ Empty screenshot {i+1}", file=sys.stderr)
            time.sleep(INTERVAL)

    if saved == 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

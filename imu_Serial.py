#!/usr/bin/env python3
"""Read IMU data and write latest Magnetometer X to a local shared JSON file."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


MAG_X_RE = re.compile(
    r"Magnetometer\s*\[uT\]\s*:\s*.*?\bx\s*=\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read IMU data and write magnetometer X to a local shared file."
    )
    parser.add_argument("--imu-port", default=None, help="IMU serial port (auto-detect if omitted)")
    parser.add_argument("--imu-baud", type=int, default=115200, help="IMU serial baud rate")
    parser.add_argument(
        "--backend",
        choices=("auto", "ybimulib", "text"),
        default="auto",
        help="IMU reader backend: auto (prefer YbImuLib), ybimulib, or text line parser",
    )
    parser.add_argument(
        "--shared-file",
        default="/tmp/imu_mag_latest.json",
        help="Local JSON file written with latest IMU value",
    )
    parser.add_argument(
        "--write-rate-hz",
        type=float,
        default=20.0,
        help="Shared-file update rate in Hz",
    )
    parser.add_argument(
        "--status-rate-hz",
        type=float,
        default=2.0,
        help="Console status print rate in Hz",
    )
    return parser.parse_args()


def choose_imu_port(requested_port: str | None, available_ports: list[object]) -> str:
    if requested_port:
        return requested_port

    acm_candidates: list[str] = []
    esp_candidates: list[str] = []
    usb_candidates: list[str] = []

    for port in available_ports:
        name = str(getattr(port, "device", "") or "")
        desc = str(getattr(port, "description", "") or "").lower()
        hwid = str(getattr(port, "hwid", "") or "").lower()
        if not name:
            continue

        if name.startswith("/dev/ttyACM"):
            acm_candidates.append(name)
            continue

        if any(
            token in desc
            for token in ("esp32", "espressif", "wch", "ch340", "ch343", "ch341", "usb jtag")
        ) or any(token in hwid for token in ("303a", "1a86")):
            esp_candidates.append(name)
            continue

        if name.startswith("/dev/ttyUSB"):
            usb_candidates.append(name)

    if acm_candidates:
        return sorted(acm_candidates)[0]
    if esp_candidates:
        return sorted(esp_candidates)[0]
    if usb_candidates:
        return sorted(usb_candidates)[0]

    if len(available_ports) == 1:
        return str(getattr(available_ports[0], "device", ""))

    available_names = [str(getattr(port, "device", "")) for port in available_ports]
    raise RuntimeError(
        "Failed to auto-detect IMU serial port. "
        f"Available ports: {available_names or ['none']}. "
        "Use --imu-port to set the port explicitly."
    )


class ImuSerialPublisher:
    def __init__(self, args: argparse.Namespace):
        serial = None
        list_ports = None
        try:
            import serial as serial_mod
            from serial.tools import list_ports as list_ports_mod

            serial = serial_mod
            list_ports = list_ports_mod
        except ImportError:
            serial = None
            list_ports = None

        self.yb_imu_cls: type[Any] | None = None
        try:
            from YbImuLib import YbImuSerial as yb_imu_cls

            self.yb_imu_cls = yb_imu_cls
        except Exception:
            self.yb_imu_cls = None

        if serial is None or list_ports is None:
            raise RuntimeError(
                "pyserial is required. Install it with: python3 -m pip install pyserial"
            )

        self.serial_mod = serial
        self.list_ports = list_ports
        self.requested_port = str(args.imu_port).strip() or None
        self.imu_baud = int(args.imu_baud)
        self.backend_mode = str(args.backend)
        self.shared_file = Path(str(args.shared_file)).expanduser()
        self.write_interval_s = 1.0 / max(0.1, float(args.write_rate_hz))
        self.status_interval_s = 1.0 / max(0.2, float(args.status_rate_hz))
        self.sample_interval_s = min(0.02, self.write_interval_s)

        self.serial = None
        self.imu_device = None
        self.active_backend = "none"
        self.serial_port: str | None = None

        self.next_connect_time = 0.0
        self.connect_backoff_s = 0.5
        self.next_sample_time = time.monotonic()
        self.next_write_time = time.monotonic()
        self.next_status_time = time.monotonic()

        self.seq = 1
        self.start_monotonic = time.monotonic()
        self.total_lines = 0
        self.parse_errors = 0
        self.write_errors = 0
        self.write_count = 0
        self.last_mag_x_ut: float | None = None
        self.last_mag_x_unix: float | None = None

        self.shared_file.parent.mkdir(parents=True, exist_ok=True)

    def _disconnect(self, reason: str) -> None:
        if self.imu_device is not None:
            for method_name in (
                "stop_receive_threading",
                "stop_receive_thread",
                "close",
            ):
                method = getattr(self.imu_device, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
            self.imu_device = None
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
        self.serial = None
        self.active_backend = "none"
        self.serial_port = None
        print(f"\nIMU serial disconnected: {reason}", flush=True)

    def _connect_ybimulib(self, port_name: str) -> None:
        if self.yb_imu_cls is None:
            raise RuntimeError("YbImuLib is not available in this Python environment")
        imu = self.yb_imu_cls(port_name, debug=False)
        create_thread = getattr(imu, "create_receive_threading", None)
        if callable(create_thread):
            create_thread()
        self.imu_device = imu
        self.active_backend = "ybimulib"

    def _connect_text(self, port_name: str) -> None:
        self.serial = self.serial_mod.Serial(
            port_name,
            self.imu_baud,
            timeout=0.05,
            write_timeout=0.2,
        )
        time.sleep(0.2)
        try:
            self.serial.reset_input_buffer()
        except Exception:
            pass
        self.active_backend = "text"

    def _try_connect(self, now_monotonic: float) -> None:
        if self.serial is not None or self.imu_device is not None:
            return
        if now_monotonic < self.next_connect_time:
            return

        available_ports = list(self.list_ports.comports())
        try:
            port_name = choose_imu_port(self.requested_port, available_ports)
            yb_error: Exception | None = None
            if self.backend_mode in {"auto", "ybimulib"}:
                try:
                    self._connect_ybimulib(port_name)
                except Exception as exc:
                    yb_error = exc
                    if self.backend_mode == "ybimulib":
                        raise
            if self.active_backend == "none" and self.backend_mode in {"auto", "text"}:
                self._connect_text(port_name)

            if self.active_backend == "none":
                raise RuntimeError(
                    f"No IMU backend available for mode={self.backend_mode}"
                    + (f" (YbImuLib error: {yb_error})" if yb_error else "")
                )
            self.serial_port = port_name
            self.connect_backoff_s = 0.5
            print(
                f"\nIMU serial connected: {port_name} @ {self.imu_baud} "
                f"backend={self.active_backend}",
                flush=True,
            )
        except Exception as exc:
            self.serial = None
            self.imu_device = None
            self.active_backend = "none"
            self.serial_port = None
            self.next_connect_time = now_monotonic + self.connect_backoff_s
            self.connect_backoff_s = min(5.0, self.connect_backoff_s * 2.0)
            print(f"\nIMU serial connect failed: {exc}", flush=True)

    def _poll_ybimulib(self) -> None:
        if self.imu_device is None:
            return
        try:
            get_magnetometer = getattr(self.imu_device, "get_magnetometer_data", None)
            if not callable(get_magnetometer):
                raise RuntimeError("YbImuLib object missing get_magnetometer_data")
            mx, _my, _mz = get_magnetometer()
            self.last_mag_x_ut = float(mx)
            self.last_mag_x_unix = time.time()
        except Exception as exc:
            self._disconnect(str(exc))

    def _poll_text(self) -> None:
        if self.serial is None:
            return
        try:
            raw = self.serial.readline()
        except Exception as exc:
            self._disconnect(str(exc))
            return

        if not raw:
            return

        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            return

        self.total_lines += 1
        if "Magnetometer" not in line or "x=" not in line:
            return

        match = MAG_X_RE.search(line)
        if match is None:
            self.parse_errors += 1
            return

        try:
            self.last_mag_x_ut = float(match.group(1))
            self.last_mag_x_unix = time.time()
        except ValueError:
            self.parse_errors += 1

    def _poll_source(self, now_monotonic: float) -> None:
        if now_monotonic < self.next_sample_time:
            return
        self.next_sample_time = now_monotonic + self.sample_interval_s
        if self.active_backend == "ybimulib":
            self._poll_ybimulib()
            return
        if self.active_backend == "text":
            self._poll_text()

    def _write_shared_file(self) -> None:
        if self.last_mag_x_ut is None:
            return
        payload = {
            "type": "imu_mag",
            "seq": self.seq,
            "ts_unix": time.time(),
            "mag_x_ut": self.last_mag_x_ut,
            "source": "imu_Serial.py",
            "valid": True,
        }
        tmp_path = self.shared_file.with_name(self.shared_file.name + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            os.replace(tmp_path, self.shared_file)
            self.seq += 1
            self.write_count += 1
        except OSError:
            self.write_errors += 1

    def _print_status(self, now_monotonic: float) -> None:
        elapsed = max(1e-6, now_monotonic - self.start_monotonic)
        write_rate = self.write_count / elapsed
        mag_text = "n/a" if self.last_mag_x_ut is None else f"{self.last_mag_x_ut:+.3f}uT"
        serial_text = self.serial_port if self.serial_port else "disconnected"
        status = (
            f"imu_port={serial_text} backend={self.active_backend} mag_x={mag_text} "
            f"file={self.shared_file} wr={write_rate:.1f}Hz parse_err={self.parse_errors} write_err={self.write_errors}"
        )
        print("\r\033[2K" + status, end="", flush=True)

    def run(self) -> int:
        print(
            "Starting IMU local writer: "
            f"file={self.shared_file} write_rate={1.0/self.write_interval_s:.1f}Hz"
        )
        print("Press Ctrl+C to stop.")

        try:
            while True:
                now = time.monotonic()
                self._try_connect(now)
                self._poll_source(now)

                while now >= self.next_write_time:
                    self._write_shared_file()
                    self.next_write_time += self.write_interval_s

                if now >= self.next_status_time:
                    self._print_status(now)
                    self.next_status_time = now + self.status_interval_s

                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nStopping IMU local writer.")
            return 0
        finally:
            if self.imu_device is not None or self.serial is not None:
                self._disconnect("shutdown")


def main() -> int:
    args = parse_args()
    try:
        app = ImuSerialPublisher(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())

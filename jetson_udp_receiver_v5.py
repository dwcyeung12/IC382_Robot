#!/usr/bin/env python3
"""v5: Receive UDP controller state and drive hardware with optional IMU/ultrasound serial telemetry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import socket
import sys
import time
from copy import deepcopy
from pathlib import Path


BUTTON_ORDER = [
    "a",
    "b",
    "x",
    "y",
    "back",
    "guide",
    "start",
    "leftstick",
    "rightstick",
    "leftshoulder",
    "rightshoulder",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
]

BUTTON_LABELS = {
    "a": "A",
    "b": "B",
    "x": "X",
    "y": "Y",
    "back": "BACK",
    "guide": "GUIDE",
    "start": "START",
    "leftstick": "L3",
    "rightstick": "R3",
    "leftshoulder": "LB",
    "rightshoulder": "RB",
    "dpad_up": "UP",
    "dpad_down": "DOWN",
    "dpad_left": "LEFT",
    "dpad_right": "RIGHT",
}

WHEEL_LABELS = ("FL", "FR", "RR", "RL")

CONTROLLER_NAME_ALIASES = {
    "xbox series x controller": "xbox",
    "xbox wireless controller": "xbox",
    "wireless controller": "pad",
    "dualsense wireless controller": "dualsense",
    "dualshock 4 wireless controller": "ds4",
}


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Use true/false."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive joystick UDP commands, drive hardware, and display optional IMU "
            "and ultrasound serial telemetry."
        )
    )
    parser.add_argument("--bind-host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=5005, help="UDP port")
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.3,
        help="Seconds before link-loss stop is triggered",
    )
    parser.add_argument(
        "--hardware",
        choices=("generic", "esp32-omni", "amd-yes-v3"),
        default="esp32-omni",
        help="Choose generic stdout output, ESP32-S3 omni drive, or v3 receiver with original AMD_YES driver-board control plus ESP32 telemetry",
    )
    parser.add_argument("--max-output", type=float, default=1.0, help="Clamp generic outputs")
    parser.add_argument(
        "--esp32-port",
        default=None,
        help="Override ESP32 serial port; omit for auto-detect in amd-yes-v3, supports /dev/ttyACM*, /dev/ttyUSB*, /dev/ttyCH343USB*",
    )
    parser.add_argument("--esp32-baud", type=int, default=115200, help="ESP32-S3 serial baud rate")
    parser.add_argument(
        "--esp32-command-deadzone",
        type=float,
        default=0.02,
        help="Deadzone used before robot velocity is sent to the ESP32",
    )
    parser.add_argument(
        "--wheel-kp",
        type=float,
        default=0.6,
        help="Per-wheel proportional gain sent to the ESP32 PID controller",
    )
    parser.add_argument(
        "--wheel-ki",
        type=float,
        default=0.0,
        help="Per-wheel integral gain sent to the ESP32 PID controller",
    )
    parser.add_argument(
        "--wheel-kd",
        type=float,
        default=0.0,
        help="Per-wheel derivative gain sent to the ESP32 PID controller",
    )
    parser.add_argument(
        "--max-wheel-rpm",
        type=float,
        default=120.0,
        help="Maximum wheel RPM target sent to the ESP32",
    )
    parser.add_argument(
        "--encoder-counts-per-rev",
        type=float,
        default=780.0,
        help="Encoder counts per wheel revolution sent to the ESP32",
    )
    parser.add_argument("--servo-port", default="/dev/ttyUSB1", help="Servo serial port")
    parser.add_argument("--servo-baud", type=int, default=115200, help="Servo baud rate")
    parser.add_argument(
        "--amd-yes-root",
        default="~/Desktop/AMD_YES",
        help="Root folder containing the AMD_YES Motor_test files for original driver-board control",
    )
    parser.add_argument("--motor-port", default="/dev/ttyUSB2", help="Override AMD_YES motor serial port")
    parser.add_argument("--motor-baud", type=int, default=115200, help="AMD_YES motor baud rate")
    parser.add_argument(
        "--motor-forward-pwm",
        type=int,
        default=2500,
        help="PWM value used for full forward wheel output on the original driver board",
    )
    parser.add_argument(
        "--motor-reverse-pwm",
        type=int,
        default=500,
        help="PWM value used for full reverse wheel output on the original driver board",
    )
    parser.add_argument(
        "--drive-threshold",
        type=float,
        default=0.35,
        help="Stick threshold for original AMD_YES discrete drive mode",
    )
    parser.add_argument(
        "--drive-deadzone",
        type=float,
        default=0.08,
        help="Deadzone for proportional omni joystick commands",
    )
    parser.add_argument(
        "--drive-scale",
        type=float,
        default=1.0,
        help="Overall joystick drive scale from 0.0 to 1.0",
    )
    parser.add_argument(
        "--servo-speed",
        type=int,
        default=800,
        help="Legacy continuous speed setting for older pair 1-3 motor-mode control",
    )
    parser.add_argument(
        "--pair-step-deg",
        type=float,
        default=45.0,
        help="Fixed step distance in degrees for servo pairs 1-3",
    )
    parser.add_argument(
        "--pair-step-time-ms",
        type=int,
        default=180,
        help="Move time for fixed position steps on servo pairs 1-3",
    )
    parser.add_argument(
        "--pair4-time-ms",
        type=int,
        default=500,
        help="Move time for pair 4 position moves",
    )
    parser.add_argument(
        "--pair4-mode1-deg",
        type=float,
        default=35.0,
        help="Pair 4 preset degree for D-pad left",
    )
    parser.add_argument(
        "--pair4-mode2-deg",
        type=float,
        default=125.0,
        help="Pair 4 preset degree for D-pad right",
    )
    parser.add_argument(
        "--imu-port",
        default="/dev/ttyUSB0",
        help="IMU serial port (use 'auto' for detection, 'none' to disable)",
    )
    parser.add_argument(
        "--imu-baud",
        type=int,
        default=115200,
        help="IMU serial baud rate",
    )
    parser.add_argument(
        "--imu-backend",
        choices=("auto", "ybimulib", "text"),
        default="auto",
        help="IMU backend: auto (prefer YbImuLib), ybimulib, or text",
    )
    parser.add_argument(
        "--imu-timeout",
        type=float,
        default=0.5,
        help="Seconds before IMU data is considered stale",
    )
    parser.add_argument(
        "--imu-turn-target-x",
        type=float,
        default=42.0,
        help="Target IMU magnetometer X for auto left-turn mode triggered by rightstick",
    )
    parser.add_argument(
        "--imu-enabled",
        type=parse_bool,
        default=True,
        help="Enable IMU serial reader (true/false)",
    )
    parser.add_argument(
        "--ultra-port",
        default="/dev/ttyCH343USB0",
        help="Ultrasound serial port (default /dev/ttyCH343USB0; use 'none' to disable)",
    )
    parser.add_argument(
        "--ultra-baud",
        type=int,
        default=115200,
        help="Ultrasound serial baud rate",
    )
    parser.add_argument(
        "--ultra-timeout",
        type=float,
        default=0.8,
        help="Seconds before ultrasound data is considered stale",
    )
    parser.add_argument(
        "--ultra-enabled",
        type=parse_bool,
        default=True,
        help="Enable ultrasound serial reader (true/false)",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help=(
            "Run a Mac-safe report-capture simulation. No UDP socket, serial port, "
            "motor, servo, IMU, or ultrasound hardware is opened."
        ),
    )
    parser.add_argument(
        "--fake-duration",
        type=float,
        default=0.0,
        help="Seconds to run fake mode before exiting; 0 runs until Ctrl+C",
    )
    parser.add_argument(
        "--fake-controller-name",
        default="Xbox Series X Controller",
        help="Controller name shown in fake status output",
    )
    parser.add_argument(
        "--fake-source-ip",
        default="10.11.31.42",
        help="Source IP shown in fake status output",
    )
    parser.add_argument(
        "--fake-missing-servos",
        default="",
        help=(
            "Comma-separated servo IDs to mark as MISS in fake mode, for example "
            "'3,7'. Empty means all fake servo pairs are OK."
        ),
    )
    return parser.parse_args()


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def clamp_range(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mix_drive(forward: float, turn: float, max_output: float) -> tuple[float, float]:
    left = clamp(forward - turn, max_output)
    right = clamp(forward + turn, max_output)
    return left, right


def format_button_state(controller_buttons: dict[str, int]) -> str:
    active = [
        BUTTON_LABELS.get(name, name)
        for name in BUTTON_ORDER
        if controller_buttons.get(name, 0)
    ]
    return ",".join(active) if active else "-"


def format_status_line(
    addr: tuple[str, int],
    controller_name: str,
    status: str,
    controller_buttons: dict[str, int],
) -> str:
    raw_name = controller_name.strip()
    name = CONTROLLER_NAME_ALIASES.get(raw_name.lower(), raw_name or "unknown")
    if len(name) > 12:
        name = name[:9] + "..."
    line = (
        f"src={addr[0]}:{addr[1]} "
        f"pad={name} | "
        f"{status} | "
        f"btn={format_button_state(controller_buttons)}"
    )
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 120
    if width > 8 and len(line) > width - 1:
        line = line[: max(0, width - 4)] + "..."
    return line


def format_wheel_values(values: list[object], precision: int = 1) -> str:
    formatted: list[str] = []
    for label, value in zip(WHEEL_LABELS, values):
        if isinstance(value, float):
            formatted.append(f"{label}={value:.{precision}f}")
        else:
            formatted.append(f"{label}={value}")
    return " ".join(formatted)


def parse_packet(message: str) -> tuple[
    float,
    float,
    list[float],
    list[int],
    list[list[int]],
    dict[str, float],
    dict[str, int],
    str,
]:
    if message.startswith("{"):
        payload = json.loads(message)
        drive = payload.get("drive", {})
        forward = float(drive.get("forward", 0.0))
        turn = float(drive.get("turn", 0.0))
        axes = [float(value) for value in payload.get("axes", [])]
        buttons = [int(value) for value in payload.get("buttons", [])]
        hats = [list(hat) for hat in payload.get("hats", [])]
        controller_axes = {
            str(key): float(value)
            for key, value in payload.get("controller_axes", {}).items()
        }
        controller_buttons = {
            str(key): int(value)
            for key, value in payload.get("controller_buttons", {}).items()
        }
        name = str(payload.get("name", "unknown"))
        return forward, turn, axes, buttons, hats, controller_axes, controller_buttons, name

    forward, turn = map(float, message.split(","))
    return forward, turn, [], [], [], {}, {}, "unknown"


def parse_ultra_sensor_line(line: str) -> dict[str, object]:
    """
    Parse ultrasound lines such as:
    Sensor 1: 23.45 cm | Sensor 2: No reading | Sensor 3: 18.20 cm | Sensor 4: 99.10 cm
    """
    result: dict[str, object] = {}
    parts = [part.strip() for part in line.split("|")]
    for part in parts:
        match = re.match(r"Sensor\s+(\d+):\s+(.+)", part, flags=re.IGNORECASE)
        if match is None:
            continue

        sensor_id = int(match.group(1))
        key = f"sensor_{sensor_id}"
        value_text = match.group(2).strip()
        if value_text.lower() == "no reading":
            result[key] = None
            continue

        value_match = re.match(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*cm",
            value_text,
            flags=re.IGNORECASE,
        )
        if value_match is None:
            result[key] = value_text
            continue
        result[key] = float(value_match.group(1))
    return result


class ImuSerialReader:
    MAG_X_RE = re.compile(
        r"Magnetometer\s*\[uT\]\s*:\s*.*?\bx\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        re.IGNORECASE,
    )

    def __init__(self, port: str, baud: int, timeout_s: float, backend: str = "auto"):
        try:
            import serial
            from serial.tools import list_ports
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyserial is required for IMU serial intake. Install it with: "
                "python3 -m pip install pyserial"
            ) from exc

        self.serial_mod = serial
        self.list_ports = list_ports
        self.yb_imu_cls = None
        try:
            from YbImuLib import YbImuSerial as yb_imu_cls

            self.yb_imu_cls = yb_imu_cls
        except Exception:
            self.yb_imu_cls = None

        self.configured_port = str(port).strip()
        self.baud = int(baud)
        self.timeout_s = max(0.05, float(timeout_s))
        self.backend_mode = str(backend).strip().lower()
        if self.backend_mode not in {"auto", "ybimulib", "text"}:
            raise RuntimeError(f"Invalid IMU backend: {backend!r}")

        self.port_obj = None
        self.imu_device = None
        self.active_backend = "none"
        self.active_port: str | None = None
        self._line_buffer = ""
        self._next_reconnect_time = 0.0
        self._reconnect_backoff_s = 0.25

        self.last_mag_x_ut: float | None = None
        self.last_imu_time: float | None = None
        self.imu_seq: int | None = None
        self.parse_errors = 0
        self.read_errors = 0

    def _is_disabled(self) -> bool:
        lowered = self.configured_port.lower()
        return lowered in {"", "none", "off", "disable", "disabled", "false", "0"}

    def _candidate_ports(self) -> list[str]:
        if self._is_disabled():
            return []
        if self.configured_port and self.configured_port.lower() != "auto":
            return [self.configured_port]

        candidates: list[tuple[int, str]] = []
        for info in self.list_ports.comports():
            device = (getattr(info, "device", "") or "").strip()
            desc = (getattr(info, "description", "") or "").strip().lower()
            if not device:
                continue
            if device == "/dev/ttyUSB0":
                score = 120
            elif device.startswith("/dev/ttyACM"):
                score = 100
            elif "imu" in desc or "yb" in desc:
                score = 90
            elif any(token in desc for token in ("esp32", "espressif", "wch", "ch340", "ch343")):
                score = 80
            elif device.startswith("/dev/ttyUSB"):
                score = 70
            else:
                score = 0
            if score > 0:
                candidates.append((score, device))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        ordered: list[str] = []
        for _, device in candidates:
            if device not in ordered:
                ordered.append(device)
        return ordered

    def _close_port(self) -> None:
        if self.imu_device is not None:
            for method_name in ("stop_receive_threading", "stop_receive_thread", "close"):
                method = getattr(self.imu_device, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
        self.imu_device = None
        if self.port_obj is not None:
            try:
                self.port_obj.close()
            except Exception:
                pass
        self.port_obj = None
        self.active_backend = "none"
        self.active_port = None
        self._line_buffer = ""

    def _connect_ybimulib(self, port: str) -> None:
        if self.yb_imu_cls is None:
            raise RuntimeError("YbImuLib is not available")
        imu = self.yb_imu_cls(port, debug=False)
        create_thread = getattr(imu, "create_receive_threading", None)
        if callable(create_thread):
            create_thread()
        self.imu_device = imu
        self.active_backend = "ybimulib"
        self.active_port = port

    def _connect_text(self, port: str) -> None:
        self.port_obj = self.serial_mod.Serial(
            port=port,
            baudrate=self.baud,
            timeout=0,
            write_timeout=0,
        )
        self.active_backend = "text"
        self.active_port = port

    def _open_if_needed(self, now: float) -> None:
        if self.active_backend != "none":
            return
        if now < self._next_reconnect_time:
            return

        for port in self._candidate_ports():
            self._close_port()
            if self.backend_mode in {"auto", "ybimulib"}:
                try:
                    self._connect_ybimulib(port)
                except Exception:
                    self._close_port()
                else:
                    self._reconnect_backoff_s = 0.25
                    return

            if self.backend_mode in {"auto", "text"}:
                try:
                    self._connect_text(port)
                except Exception:
                    self._close_port()
                    continue
                self._reconnect_backoff_s = 0.25
                return

        self._close_port()
        self._next_reconnect_time = now + self._reconnect_backoff_s
        self._reconnect_backoff_s = min(3.0, self._reconnect_backoff_s * 1.8)

    def _mark_io_failure(self, now: float) -> None:
        self.read_errors += 1
        self._close_port()
        self._next_reconnect_time = now + self._reconnect_backoff_s
        self._reconnect_backoff_s = min(3.0, self._reconnect_backoff_s * 1.8)

    def _handle_line(self, line: str, now: float) -> None:
        match = self.MAG_X_RE.search(line)
        if match is None:
            if "Magnetometer" in line and "x=" in line:
                self.parse_errors += 1
            return
        try:
            mag_value = float(match.group(1))
        except (TypeError, ValueError):
            self.parse_errors += 1
            return

        self.last_mag_x_ut = mag_value
        self.last_imu_time = now
        self.imu_seq = 1 if self.imu_seq is None else (self.imu_seq + 1)

    def _poll_ybimulib(self, now: float) -> None:
        if self.imu_device is None:
            return
        try:
            getter = getattr(self.imu_device, "get_magnetometer_data", None)
            if not callable(getter):
                raise RuntimeError("YbImuLib object missing get_magnetometer_data")
            mx, _my, _mz = getter()
            self.last_mag_x_ut = float(mx)
            self.last_imu_time = now
            self.imu_seq = 1 if self.imu_seq is None else (self.imu_seq + 1)
        except Exception:
            self._mark_io_failure(now)

    def _poll_text(self, now: float) -> None:
        if self.port_obj is None:
            return

        try:
            waiting = getattr(self.port_obj, "in_waiting", 0) or 0
            if waiting <= 0:
                return
            chunk = self.port_obj.read(waiting)
        except Exception:
            self._mark_io_failure(now)
            return

        if not chunk:
            return

        text = self._line_buffer + chunk.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        complete_lines: list[str] = []
        carry = ""
        for item in lines:
            if item.endswith("\n") or item.endswith("\r"):
                complete_lines.append(item.rstrip("\r\n"))
            else:
                carry = item
        self._line_buffer = carry

        for line in complete_lines:
            self._handle_line(line, now)

    def poll(self, now: float) -> None:
        self._open_if_needed(now)
        if self.active_backend == "ybimulib":
            self._poll_ybimulib(now)
            return
        if self.active_backend == "text":
            self._poll_text(now)
            return

    def current_mag_x_ut(self, now: float, fresh_only: bool = True) -> float | None:
        if self.last_mag_x_ut is None or self.last_imu_time is None:
            return None
        if fresh_only and (now - self.last_imu_time) > self.timeout_s:
            return None
        return self.last_mag_x_ut

    def status_text(self, now: float) -> str:
        if self._is_disabled():
            return "imu=(disabled)"
        if self.last_imu_time is None or self.last_mag_x_ut is None:
            if self.active_port is not None:
                return f"imu=(waiting port={self.active_port} backend={self.active_backend})"
            return "imu=(waiting)"
        age = now - self.last_imu_time
        if age > self.timeout_s:
            return f"imu=(stale age={age:.2f}s)"
        return f"imu=(x={self.last_mag_x_ut:+.3f}uT age={age:.2f}s)"

    def source_text(self) -> str:
        if self._is_disabled():
            return "IMU telemetry: disabled by imu-port"
        if self.active_port is not None:
            return (
                f"Reading IMU serial on {self.active_port} @ {self.baud} "
                f"backend={self.active_backend}"
            )
        return (
            f"Reading IMU serial (reconnecting) configured={self.configured_port} "
            f"@ {self.baud} backend={self.backend_mode}"
        )

    def close(self) -> None:
        self._close_port()


class UltraSerialReader:
    def __init__(self, port: str, baud: int, timeout_s: float):
        try:
            import serial
            from serial.tools import list_ports
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyserial is required for ultrasound serial intake. Install it with: "
                "python3 -m pip install pyserial"
            ) from exc

        self.serial_mod = serial
        self.list_ports = list_ports

        self.configured_port = str(port).strip()
        self.baud = int(baud)
        self.timeout_s = max(0.05, float(timeout_s))

        self.port_obj = None
        self.active_port: str | None = None
        self._line_buffer = ""
        self._next_reconnect_time = 0.0
        self._reconnect_backoff_s = 0.25

        self.last_ultra_time: float | None = None
        self.ultra_seq: int | None = None
        self.sensor_values: dict[str, object] = {}
        self.parse_errors = 0
        self.read_errors = 0

    def _is_disabled(self) -> bool:
        lowered = self.configured_port.lower()
        return lowered in {"", "none", "off", "disable", "disabled", "false", "0"}

    def _candidate_ports(self) -> list[str]:
        if self._is_disabled():
            return []
        if self.configured_port and self.configured_port.lower() != "auto":
            return [self.configured_port]

        candidates: list[tuple[int, str]] = []
        for info in self.list_ports.comports():
            device = (getattr(info, "device", "") or "").strip()
            desc = (getattr(info, "description", "") or "").strip().lower()
            if not device:
                continue
            if device == "/dev/ttyCH343USB0":
                score = 120
            elif "ultra" in desc or "sonar" in desc or "distance" in desc:
                score = 110
            elif device.startswith("/dev/ttyCH343USB"):
                score = 100
            elif any(token in desc for token in ("wch", "ch343", "ch340")):
                score = 90
            elif device.startswith("/dev/ttyUSB"):
                score = 80
            else:
                score = 0
            if score > 0:
                candidates.append((score, device))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        ordered: list[str] = []
        for _, device in candidates:
            if device not in ordered:
                ordered.append(device)
        return ordered

    def _close_port(self) -> None:
        if self.port_obj is not None:
            try:
                self.port_obj.close()
            except Exception:
                pass
        self.port_obj = None
        self.active_port = None
        self._line_buffer = ""

    def _open_if_needed(self, now: float) -> None:
        if self.port_obj is not None:
            return
        if now < self._next_reconnect_time:
            return

        for port in self._candidate_ports():
            self._close_port()
            try:
                self.port_obj = self.serial_mod.Serial(
                    port=port,
                    baudrate=self.baud,
                    timeout=0,
                    write_timeout=0,
                )
            except Exception:
                self._close_port()
                continue
            self.active_port = port
            self._reconnect_backoff_s = 0.25
            return

        self._close_port()
        self._next_reconnect_time = now + self._reconnect_backoff_s
        self._reconnect_backoff_s = min(3.0, self._reconnect_backoff_s * 1.8)

    def _mark_io_failure(self, now: float) -> None:
        self.read_errors += 1
        self._close_port()
        self._next_reconnect_time = now + self._reconnect_backoff_s
        self._reconnect_backoff_s = min(3.0, self._reconnect_backoff_s * 1.8)

    def _handle_line(self, line: str, now: float) -> None:
        if not line.startswith("Sensor "):
            return
        parsed = parse_ultra_sensor_line(line)
        if not parsed:
            self.parse_errors += 1
            return
        self.sensor_values = parsed
        self.last_ultra_time = now
        self.ultra_seq = 1 if self.ultra_seq is None else (self.ultra_seq + 1)

    def poll(self, now: float) -> None:
        self._open_if_needed(now)
        if self.port_obj is None:
            return

        try:
            waiting = getattr(self.port_obj, "in_waiting", 0) or 0
            if waiting <= 0:
                return
            chunk = self.port_obj.read(waiting)
        except Exception:
            self._mark_io_failure(now)
            return

        if not chunk:
            return

        text = self._line_buffer + chunk.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        complete_lines: list[str] = []
        carry = ""
        for item in lines:
            if item.endswith("\n") or item.endswith("\r"):
                complete_lines.append(item.rstrip("\r\n"))
            else:
                carry = item
        self._line_buffer = carry

        for line in complete_lines:
            self._handle_line(line, now)

    def _summary_text(self) -> str:
        if not self.sensor_values:
            return "no-data"
        keys = sorted(
            self.sensor_values.keys(),
            key=lambda item: int(item.split("_", 1)[1]) if item.startswith("sensor_") else 999,
        )
        parts: list[str] = []
        for key in keys:
            value = self.sensor_values.get(key)
            sensor_suffix = key.split("_", 1)[1] if "_" in key else key
            if value is None:
                parts.append(f"s{sensor_suffix}=NR")
            elif isinstance(value, float):
                parts.append(f"s{sensor_suffix}={value:.1f}cm")
            else:
                parts.append(f"s{sensor_suffix}={value}")
        return " ".join(parts)

    def status_text(self, now: float) -> str:
        if self._is_disabled():
            return "ultra_serial=(disabled)"
        if self.last_ultra_time is None or not self.sensor_values:
            if self.active_port is not None:
                return f"ultra_serial=(waiting port={self.active_port})"
            return "ultra_serial=(waiting)"
        age = now - self.last_ultra_time
        if age > self.timeout_s:
            return f"ultra_serial=(stale age={age:.2f}s)"
        return f"ultra_serial=({self._summary_text()} age={age:.2f}s)"

    def source_text(self) -> str:
        if self._is_disabled():
            return "Ultrasound telemetry: disabled by ultra-port"
        if self.active_port is not None:
            return f"Reading ultrasound serial on {self.active_port} @ {self.baud}"
        return (
            f"Reading ultrasound serial (reconnecting) configured={self.configured_port} "
            f"@ {self.baud}"
        )

    def close(self) -> None:
        self._close_port()


class GenericDriveController:
    def send_motor_command(self, left: float, right: float) -> None:
        print(f"\rleft={left:+.3f} right={right:+.3f}", end="", flush=True)

    def poll(self) -> None:
        return

    def on_timeout(self) -> None:
        self.send_motor_command(0.0, 0.0)

    def close(self) -> None:
        self.on_timeout()
        print()


def load_local_servo_module():
    search_paths = [
        Path(__file__).with_name("servo_pair_controller_V2.remote.py"),
        Path(__file__).resolve().parent.parent / "servo_pair_controller_V2.remote.py",
        Path(__file__).resolve().parent / "Motor_test" / "servo_pair_controller_V2.py",
        Path(__file__).resolve().parent.parent / "Motor_test" / "servo_pair_controller_V2.py",
    ]
    checked_paths: list[str] = []
    for module_path in search_paths:
        checked_paths.append(str(module_path))
        if not module_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "servo_pair_controller_v2_remote", module_path
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, module)
        spec.loader.exec_module(module)
        return module
    raise FileNotFoundError(
        "Failed to find servo helper module. Checked: " + ", ".join(checked_paths)
    )


def load_local_v2_receiver_module():
    search_paths = [
        Path(__file__).with_name("jetson_udp_receiver_v2.py"),
        Path(__file__).resolve().parent.parent / "jetson_udp_receiver_v2.py",
    ]
    checked_paths: list[str] = []
    for module_path in search_paths:
        checked_paths.append(str(module_path))
        if not module_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "jetson_udp_receiver_v2_local", module_path
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, module)
        spec.loader.exec_module(module)
        return module
    raise FileNotFoundError(
        "Failed to find jetson_udp_receiver_v2.py. Checked: " + ", ".join(checked_paths)
    )


class AmdYesTelemetryV3Controller:
    def __init__(self, args: argparse.Namespace):
        try:
            import serial
            from serial.tools import list_ports
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for --hardware amd-yes-v3. Install it with: "
                "python3 -m pip install pyserial"
            ) from exc

        base_args = argparse.Namespace(**vars(args))
        servo_disabled = str(getattr(base_args, "servo_port", "")).strip().lower() in {
            "",
            "none",
            "off",
            "disabled",
            "null",
        }
        if servo_disabled:
            base_args.servo_port = None
        setattr(base_args, "pair_hold_speed", getattr(base_args, "pair_hold_speed", 300))
        setattr(
            base_args,
            "pair_hold_deadband_pulses",
            getattr(base_args, "pair_hold_deadband_pulses", 64),
        )
        if servo_disabled:
            self.base = AmdYesDriveOnlyController(base_args)
        else:
            v2_module = load_local_v2_receiver_module()
            self.base = v2_module.AmdYesController(base_args)
        self.serial_mod = serial
        self.list_ports = list_ports
        self.telemetry: dict[str, object] = {}
        self.telemetry_cfg: str | None = None
        self.available_ports = list(self.list_ports.comports())
        self.esp32 = None
        raw_esp32_port = getattr(args, "esp32_port", None)
        esp32_port_text = str(raw_esp32_port or "").strip()
        telemetry_disabled = esp32_port_text.lower() in {
            "none",
            "off",
            "disabled",
            "null",
        }
        if not telemetry_disabled:
            available_port_names = [port.device for port in self.available_ports]
            exclude_ports = {
                str(args.motor_port or "").strip(),
                str(args.servo_port or "").strip(),
            }
            requested_port = esp32_port_text or None
            try:
                esp32_port = self._resolve_esp32_port(requested_port, available_port_names, exclude_ports)
                self.esp32 = serial.Serial(
                    esp32_port,
                    int(args.esp32_baud),
                    timeout=0.02,
                    write_timeout=0.2,
                )
                time.sleep(2.0)
                self._drain_esp32_input()
                self._write_esp32_line("STREAM ON")
                self._write_esp32_line("STATUS")
                print(
                    "AMD_YES v3 ready: "
                    f"motor_driver=original_v2_driver_board telemetry_port={esp32_port}"
                )
            except Exception:
                if requested_port is not None:
                    raise
                self.esp32 = None
                print(
                    "AMD_YES v3 ready: "
                    "motor_driver=original_v2_driver_board telemetry=auto_not_found"
                )
        else:
            print("AMD_YES v3 ready: motor_driver=original_v2_driver_board telemetry=disabled")

    def _resolve_esp32_port(
        self,
        requested_port: str | None,
        available_port_names: list[str],
        exclude_ports: set[str],
    ) -> str:
        if requested_port and requested_port in available_port_names:
            return requested_port

        preferred_candidates: list[str] = []
        fallback_candidates: list[str] = []
        for port in self.available_ports:
            name = port.device
            desc = (port.description or "").lower()
            hwid = (port.hwid or "").lower()
            if name in exclude_ports:
                continue
            if name.startswith("/dev/ttyACM") or "usbmodem" in name:
                preferred_candidates.append(name)
                continue
            if "espressif" in desc or "esp32" in desc or "usb jtag" in desc or "303a" in hwid:
                preferred_candidates.append(name)
                continue
            if name.startswith("/dev/ttyUSB") or name.startswith("/dev/ttyCH343USB"):
                fallback_candidates.append(name)
                continue
            if (
                "wch" in desc
                or "usb single serial" in desc
                or "ch340" in desc
                or "ch341" in desc
                or "ch343" in desc
                or "1a86" in hwid
            ):
                fallback_candidates.append(name)

        for candidate in preferred_candidates + fallback_candidates:
            if candidate not in exclude_ports:
                return candidate
        if requested_port:
            return requested_port
        raise RuntimeError(
            "Failed to auto-detect ESP32 telemetry serial port. "
            f"Available ports: {available_port_names or ['none']}"
        )

    def _write_esp32_line(self, line: str) -> None:
        if self.esp32 is None:
            return
        self.esp32.write((line + "\n").encode("utf-8"))
        self.esp32.flush()

    def _drain_esp32_input(self) -> None:
        if self.esp32 is None:
            return
        try:
            waiting = getattr(self.esp32, "in_waiting", 0)
            if waiting:
                self.esp32.read(waiting)
        except Exception:
            pass

    def _parse_float_list(self, value: str) -> list[float]:
        return [float(item) for item in value.split(",") if item]

    def _parse_int_list(self, value: str) -> list[int]:
        return [int(item) for item in value.split(",") if item]

    def _handle_telemetry_line(self, line: str) -> None:
        if not line:
            return
        if line.startswith("CFG "):
            self.telemetry_cfg = line
            return
        if line.startswith("TEL "):
            parsed: dict[str, object] = {}
            for item in line.split()[1:]:
                if "=" not in item:
                    continue
                key, raw = item.split("=", 1)
                if key == "ultra_cm":
                    parsed[key] = float(raw) if raw != "nan" else float("nan")
                elif key == "rpm":
                    parsed[key] = self._parse_float_list(raw)
                elif key == "count":
                    parsed[key] = self._parse_int_list(raw)
            self.telemetry = parsed
            return
        if line.startswith(("INFO", "FAULT", "WARN", "OK")):
            print(f"\nESP32 {line}", flush=True)

    def poll(self) -> None:
        if self.esp32 is None:
            return
        while True:
            try:
                waiting = getattr(self.esp32, "in_waiting", 0)
                if waiting <= 0:
                    break
                line = self.esp32.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    break
                self._handle_telemetry_line(line)
            except Exception:
                break

    def _telemetry_summary(self) -> str:
        if self.esp32 is None:
            return "tel=(disabled)"
        if not self.telemetry:
            return "tel=(waiting)"
        ultra = self.telemetry.get("ultra_cm", float("nan"))
        ultra_text = "nan" if ultra != ultra else f"{float(ultra):.1f}"
        rpm = self.telemetry.get("rpm", [])
        count = self.telemetry.get("count", [])
        if isinstance(rpm, list) and len(rpm) == 4 and isinstance(count, list) and len(count) == 4:
            rpm_text = format_wheel_values([round(float(v), 1) for v in rpm], precision=1)
            count_text = format_wheel_values(count, precision=0)
            return (
                "tel=("
                f"ultra={ultra_text} "
                f"rpm={rpm_text} "
                f"count={count_text}"
                ")"
            )
        return f"tel=(ultra={ultra_text})"

    def handle_packet(
        self,
        controller_axes: dict[str, float],
        controller_buttons: dict[str, int],
        imu_mag_x_ut: float | None = None,
    ) -> str:
        if isinstance(self.base, AmdYesDriveOnlyController):
            base_status = self.base.handle_packet(
                controller_axes,
                controller_buttons,
                imu_mag_x_ut=imu_mag_x_ut,
            )
        else:
            base_status = self.base.handle_packet(controller_axes, controller_buttons)
        if self.esp32 is None:
            return base_status
        return f"{base_status} {self._telemetry_summary()}"

    def on_timeout(self) -> None:
        self.base.on_timeout()

    def close(self) -> None:
        try:
            self.base.close()
        finally:
            if self.esp32 is not None:
                self.esp32.close()


class AmdYesDriveOnlyController:
    def __init__(self, args: argparse.Namespace):
        import importlib

        root = Path(args.amd_yes_root).expanduser()
        motor_test_dir = root / "Motor_test"
        if not motor_test_dir.is_dir():
            raise FileNotFoundError(f"AMD_YES Motor_test folder not found: {motor_test_dir}")

        sys.path.insert(0, str(motor_test_dir))
        self.omni = importlib.import_module("omni_car_control")
        from serial.tools import list_ports

        self.available_ports = list(list_ports.comports())
        available_port_names = [port.device for port in self.available_ports]
        requested_motor_port = args.motor_port or getattr(self.omni, "DEFAULT_MOTOR_PORT", "/dev/motor")
        motor_port, self.motor_bus = self._connect_motor_bus(
            requested_port=requested_motor_port,
            baud=args.motor_baud,
            excluded_ports=self._reserved_port_names(
                getattr(args, "imu_port", None),
                getattr(args, "esp32_port", None),
            ),
        )
        if motor_port != requested_motor_port:
            print(
                f"Requested motor port {requested_motor_port} was unavailable; "
                f"using detected Bus370 port {motor_port}."
            )

        self.drive_deadzone = max(0.0, min(0.5, args.drive_deadzone))
        self.drive_scale = max(0.0, min(1.0, args.drive_scale))
        self.drive_threshold = max(0.05, min(0.95, args.drive_threshold))
        self.motor_forward_pwm = max(1500, min(2500, args.motor_forward_pwm))
        self.motor_reverse_pwm = max(500, min(1500, args.motor_reverse_pwm))
        self.imu_turn_target_x = float(getattr(args, "imu_turn_target_x", 42.0))
        self.imu_turn_active = False
        self.last_buttons = {name: 0 for name in BUTTON_ORDER}
        print(
            "AMD_YES drive-only ready: "
            f"motor_port={motor_port} servo=disabled"
        )

    def _resolve_motor_port(
        self,
        requested_port: str,
        available_port_names: list[str],
    ) -> str:
        if requested_port in available_port_names:
            return requested_port
        for port in self.available_ports:
            if port.vid == 0x1A86 and port.pid == 0x7523:
                return port.device
        if len(available_port_names) == 1:
            return available_port_names[0]
        return requested_port

    def _reserved_port_names(self, *ports: object) -> set[str]:
        reserved: set[str] = set()
        for port in ports:
            text = str(port or "").strip()
            if not text:
                continue
            if text.lower() in {"none", "off", "disabled", "null", "auto"}:
                continue
            reserved.add(text)
        return reserved

    def _motor_candidate_ports(self, requested_port: str, excluded_ports: set[str]) -> list[str]:
        candidates: list[str] = []

        def add_candidate(port_name: str) -> None:
            port_text = str(port_name or "").strip()
            if not port_text or port_text in excluded_ports or port_text in candidates:
                return
            candidates.append(port_text)

        add_candidate(requested_port)
        for port in sorted(self.available_ports, key=lambda item: item.device or ""):
            if port.vid == 0x1A86 and port.pid == 0x7523:
                add_candidate(port.device)
        for port_name in sorted(
            port.device for port in self.available_ports if (port.device or "").startswith("/dev/ttyUSB")
        ):
            add_candidate(port_name)
        return candidates

    def _probe_bus370_port(self, port_name: str, baud: int) -> bool:
        import serial

        probe = serial.Serial(port_name, baud, timeout=0.35, write_timeout=0.35)
        try:
            time.sleep(0.12)
            probe.reset_input_buffer()
            probe.reset_output_buffer()
            for motor_id in range(5):
                probe.write(f"#{motor_id:03d}PID!".encode("ascii"))
                probe.flush()
                time.sleep(0.05)
                if probe.read(64):
                    return True
            return False
        finally:
            probe.close()

    def _connect_motor_bus(
        self,
        requested_port: str,
        baud: int,
        excluded_ports: set[str],
    ) -> tuple[str, object]:
        available_port_names = [port.device for port in self.available_ports]
        errors: list[str] = []
        fallback_port = self._resolve_motor_port(requested_port, available_port_names)
        for port_name in self._motor_candidate_ports(fallback_port, excluded_ports):
            try:
                if not self._probe_bus370_port(port_name, int(baud)):
                    errors.append(f"{port_name}: no Bus370 ping response")
                    continue
            except Exception as exc:
                errors.append(f"{port_name}: probe failed ({exc})")
                continue

            last_exc: Exception | None = None
            for _attempt in range(3):
                try:
                    return (
                        port_name,
                        self.omni.Bus370(
                            port_name,
                            baud=baud,
                            timeout=1.2,
                            debug=False,
                        ),
                    )
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.15)
            errors.append(f"{port_name}: open failed ({last_exc})")

        joined_errors = "; ".join(errors) if errors else "no candidate ports"
        raise RuntimeError(
            f"Failed to find a responsive motor controller. "
            f"Requested={requested_port!r} excluded={sorted(excluded_ports)} "
            f"available={available_port_names or ['none']} details={joined_errors}"
        )

    def _rising_edge(self, controller_buttons: dict[str, int], name: str) -> bool:
        return bool(controller_buttons.get(name, 0)) and not bool(self.last_buttons.get(name, 0))

    def _emergency_stop_pressed(self, controller_buttons: dict[str, int]) -> bool:
        if self._rising_edge(controller_buttons, "guide"):
            return True
        back_pressed = bool(controller_buttons.get("back", 0))
        start_pressed = bool(controller_buttons.get("start", 0))
        if back_pressed and start_pressed:
            return self._rising_edge(controller_buttons, "back") or self._rising_edge(
                controller_buttons, "start"
            )
        return False

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.drive_deadzone else value

    def stop_drive(self) -> None:
        self.omni.stop_all(self.motor_bus)

    def emergency_stop(self) -> None:
        self.imu_turn_active = False
        self.stop_drive()

    def _apply_discrete_drive(self, fl: int, fr: int, rr: int, rl: int) -> None:
        self.omni.apply_drive(
            self.motor_bus,
            self.motor_forward_pwm,
            self.motor_reverse_pwm,
            fl,
            fr,
            rr,
            rl,
        )

    def _drive_rotate_left(self) -> None:
        self._apply_discrete_drive(
            -self.omni.DIR_FL,
            self.omni.DIR_FR,
            self.omni.DIR_RR,
            -self.omni.DIR_RL,
        )

    def drive_omni(self, controller_axes: dict[str, float]) -> tuple[float, float, float, str]:
        strafe = self._apply_deadzone(controller_axes.get("leftx", 0.0)) * self.drive_scale
        forward = -self._apply_deadzone(controller_axes.get("lefty", 0.0)) * self.drive_scale
        rotate = self._apply_deadzone(controller_axes.get("rightx", 0.0)) * self.drive_scale

        if strafe == 0.0 and forward == 0.0 and rotate == 0.0:
            self.stop_drive()
            return 0.0, 0.0, 0.0, "stop"

        threshold = self.drive_threshold
        abs_forward = abs(forward)
        abs_strafe = abs(strafe)
        abs_rotate = abs(rotate)

        if abs_forward >= abs_strafe and abs_forward >= abs_rotate and abs_forward >= threshold:
            if forward > 0:
                self._apply_discrete_drive(
                    self.omni.DIR_FL,
                    self.omni.DIR_FR,
                    self.omni.DIR_RR,
                    self.omni.DIR_RL,
                )
                return strafe, forward, rotate, "forward"
            self._apply_discrete_drive(
                -self.omni.DIR_FL,
                -self.omni.DIR_FR,
                -self.omni.DIR_RR,
                -self.omni.DIR_RL,
            )
            return strafe, forward, rotate, "reverse"

        if abs_strafe >= abs_rotate and abs_strafe >= threshold:
            if strafe > 0:
                self._apply_discrete_drive(
                    self.omni.DIR_FL,
                    -self.omni.DIR_FR,
                    self.omni.DIR_RR,
                    -self.omni.DIR_RL,
                )
                return strafe, forward, rotate, "strafe_right"
            self._apply_discrete_drive(
                -self.omni.DIR_FL,
                self.omni.DIR_FR,
                -self.omni.DIR_RR,
                self.omni.DIR_RL,
            )
            return strafe, forward, rotate, "strafe_left"

        if abs_rotate >= threshold:
            if rotate > 0:
                self._apply_discrete_drive(
                    self.omni.DIR_FL,
                    -self.omni.DIR_FR,
                    -self.omni.DIR_RR,
                    self.omni.DIR_RL,
                )
                return strafe, forward, rotate, "rotate_right"
            self._apply_discrete_drive(
                -self.omni.DIR_FL,
                self.omni.DIR_FR,
                self.omni.DIR_RR,
                -self.omni.DIR_RL,
            )
            return strafe, forward, rotate, "rotate_left"

        self.stop_drive()
        return strafe, forward, rotate, "stop"

    def handle_packet(
        self,
        controller_axes: dict[str, float],
        controller_buttons: dict[str, int],
        imu_mag_x_ut: float | None = None,
    ) -> str:
        if self._emergency_stop_pressed(controller_buttons):
            print("\nEmergency stop.")
            self.emergency_stop()
            self.imu_turn_active = False
            self.last_buttons = dict(controller_buttons)
            return "drv=stopped"

        if self._rising_edge(controller_buttons, "rightstick"):
            self.imu_turn_active = True
            current_text = "n/a" if imu_mag_x_ut is None else f"{imu_mag_x_ut:+.3f}"
            print(
                "\nIMU left-turn armed: "
                f"target_x={self.imu_turn_target_x:.3f} current_x={current_text}"
            )

        if self.imu_turn_active:
            if imu_mag_x_ut is None:
                self.stop_drive()
                strafe, forward, rotate, drive_mode = (0.0, 0.0, 0.0, "imu_turn_waiting")
            elif imu_mag_x_ut >= self.imu_turn_target_x:
                self.stop_drive()
                self.imu_turn_active = False
                strafe, forward, rotate, drive_mode = (0.0, 0.0, 0.0, "imu_turn_done")
            else:
                self._drive_rotate_left()
                strafe, forward, rotate, drive_mode = (0.0, 0.0, -1.0, "imu_turn_left")
        else:
            strafe, forward, rotate, drive_mode = self.drive_omni(controller_axes)

        if self._rising_edge(controller_buttons, "back"):
            print("\nDrive stop.")
            self.stop_drive()
            self.imu_turn_active = False
            strafe = 0.0
            forward = 0.0
            rotate = 0.0
            drive_mode = "stop"

        self.last_buttons = dict(controller_buttons)
        if drive_mode.startswith("imu_turn"):
            imu_text = "n/a" if imu_mag_x_ut is None else f"{imu_mag_x_ut:+.3f}"
            return (
                f"drv={drive_mode} imu_x={imu_text} target={self.imu_turn_target_x:+.3f} "
                f"x={strafe:+.2f} y={forward:+.2f} rot={rotate:+.2f}"
            )
        return f"drv={drive_mode} x={strafe:+.2f} y={forward:+.2f} rot={rotate:+.2f}"

    def poll(self) -> None:
        return

    def on_timeout(self) -> None:
        self.imu_turn_active = False
        self.emergency_stop()

    def close(self) -> None:
        try:
            self.emergency_stop()
        finally:
            self.motor_bus.close()


class Esp32S3OmniController:
    VELOCITY_RESEND_INTERVAL_S = 0.03
    VELOCITY_CHANGE_EPSILON = 0.005
    STOP_RESEND_INTERVAL_S = 0.08

    def __init__(self, args: argparse.Namespace):
        try:
            import serial
            from serial.tools import list_ports
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for --hardware esp32-omni. Install it with: "
                "python3 -m pip install pyserial"
            ) from exc

        self.serial_mod = serial
        self.list_ports = list_ports
        self.servo_enabled = str(args.servo_port).strip().lower() not in {
            "",
            "none",
            "off",
            "disabled",
            "null",
        }
        self.servo_mod = load_local_servo_module() if self.servo_enabled else None
        self.drive_deadzone = max(0.0, min(0.5, args.drive_deadzone))
        self.drive_scale = max(0.0, min(1.0, args.drive_scale))
        self.command_deadzone = max(0.0, min(0.5, args.esp32_command_deadzone))
        self.servo_speed = max(0, min(1000, int(args.servo_speed)))
        self.pair_step_deg = max(0.24, min(3600.0, float(args.pair_step_deg)))
        self.pair_step_time_ms = max(0, min(30000, int(args.pair_step_time_ms)))
        self.pair4_time_ms = max(0, min(30000, int(args.pair4_time_ms)))
        self.pair4_mode1_deg = float(args.pair4_mode1_deg)
        self.pair4_mode2_deg = float(args.pair4_mode2_deg)
        self.wheel_kp = float(args.wheel_kp)
        self.wheel_ki = float(args.wheel_ki)
        self.wheel_kd = float(args.wheel_kd)
        self.max_wheel_rpm = float(args.max_wheel_rpm)
        self.encoder_counts_per_rev = float(args.encoder_counts_per_rev)
        self.telemetry: dict[str, object] = {}
        self.last_sent_vel = (0.0, 0.0, 0.0)
        self.last_velocity_send_time = 0.0
        self.last_stop_sent = True
        self.pending_velocity_seq: int | None = None
        self.pending_velocity = (0.0, 0.0, 0.0)
        self.last_acked_velocity = (0.0, 0.0, 0.0)
        self.last_ack_time = 0.0
        self.next_velocity_seq = 1
        self.last_buttons = {name: 0 for name in BUTTON_ORDER}
        self.last_servo_motion: tuple[int, int] | None = None
        self.pair_targets: dict[int, int] = {}
        self.position_read_warned_ids: set[int] = set()
        self.active_step_motion: dict[str, object] | None = None

        self.servo = None
        self.valid_pairs: list[tuple[int, int, int]] = []
        self.selected_pair_index = 0
        self.pair4_available = False
        self.fixed_step_ready = False
        self.distance_step_ready = False
        self.pair_control_mode = "disabled"
        if self.servo_enabled:
            assert self.servo_mod is not None
            self.servo = self.servo_mod.BusServo(
                port=args.servo_port,
                baud=args.servo_baud,
                timeout=0.2,
            )
            self.valid_pairs = self.servo_mod.verify_pairs(self.servo)
            if not self.valid_pairs:
                self.servo.close()
                raise RuntimeError("No valid servo pairs found.")

            self.pair4_available = any(
                primary == self.servo_mod.PAIR4_PRIMARY
                and opposite == self.servo_mod.PAIR4_OPPOSITE
                for _, primary, opposite in self.valid_pairs
            )
            self.fixed_step_ready = self._check_fixed_step_support()
            self.distance_step_ready = self._check_distance_step_support()
            self.pair_control_mode = (
                "step" if (self.fixed_step_ready or self.distance_step_ready) else "motor"
            )
            if self.pair4_available:
                self._ensure_pair4_servo_mode()

        self.available_ports = list(self.list_ports.comports())
        available_port_names = [port.device for port in self.available_ports]
        esp32_port = self._resolve_esp32_port(args.esp32_port, available_port_names, args.servo_port)
        self.esp32 = serial.Serial(
            esp32_port,
            int(args.esp32_baud),
            timeout=0.02,
            write_timeout=0.2,
        )
        time.sleep(2.0)
        self._drain_esp32_input()
        self._send_config()
        self.stop_drive()
        if self.servo_enabled:
            self.stop_all_pairs_1_to_3()

        print(
            "ESP32-S3 omni drive ready: "
            f"esp32_port={esp32_port} servo_port={args.servo_port} "
            f"servo_enabled={self.servo_enabled} "
            f"selected_pair={(self.selected_pair_index + 1) if self.servo_enabled else 'n/a'}"
        )
        if self.servo_enabled:
            print(
                "Mapping: left stick = proportional strafe/forward, right stick X = proportional rotation, "
                "A/B/X = select pair 1/2/3, Y = toggle pair mode, "
                "left/right stick click = cycle pairs, LB/RB = move selected pair, "
                "D-pad left/right = pair4 presets, D-pad up/down = pair step size, "
                "Start = stop selected pair, Back = stop drive, Guide = emergency stop all."
            )
        else:
            print(
                "Mapping: left stick = proportional strafe/forward, right stick X = proportional rotation, "
                "Back = stop drive, Guide = emergency stop all. Servo controls are disabled."
            )

    def _resolve_esp32_port(
        self,
        requested_port: str | None,
        available_port_names: list[str],
        servo_port: str,
    ) -> str:
        if requested_port and requested_port in available_port_names:
            return requested_port

        preferred_candidates: list[str] = []
        fallback_candidates: list[str] = []
        for port in self.available_ports:
            name = port.device
            desc = (port.description or "").lower()
            hwid = (port.hwid or "").lower()
            if name == servo_port:
                continue
            if name.startswith("/dev/ttyACM"):
                preferred_candidates.append(name)
                continue
            if "usbmodem" in name:
                preferred_candidates.append(name)
                continue
            if "espressif" in desc or "esp32" in desc or "usb jtag" in desc or "303a" in hwid:
                preferred_candidates.append(name)
                continue
            if name.startswith("/dev/ttyUSB"):
                fallback_candidates.append(name)
                continue
            if (
                "wch" in desc
                or "usb single serial" in desc
                or "ch340" in desc
                or "ch341" in desc
                or "1a86" in hwid
            ):
                fallback_candidates.append(name)

        for candidate in preferred_candidates + fallback_candidates:
            if candidate != servo_port:
                return candidate
        if requested_port:
            return requested_port
        raise RuntimeError(
            "Failed to auto-detect ESP32-S3 serial port. "
            f"Available ports: {available_port_names or ['none']}"
        )

    def _write_esp32_line(self, line: str) -> None:
        self.esp32.write((line + "\n").encode("utf-8"))
        self.esp32.flush()

    def _drain_esp32_input(self) -> None:
        try:
            waiting = getattr(self.esp32, "in_waiting", 0)
            if waiting:
                self.esp32.read(waiting)
        except Exception:
            pass

    def _send_config(self) -> None:
        self._write_esp32_line(f"PID ALL {self.wheel_kp:.4f} {self.wheel_ki:.4f} {self.wheel_kd:.4f}")
        self._write_esp32_line(f"MAXRPM {self.max_wheel_rpm:.2f}")
        self._write_esp32_line(f"CPR {self.encoder_counts_per_rev:.2f}")
        self._write_esp32_line("STATUS")

    def _parse_float_list(self, value: str) -> list[float]:
        return [float(item) for item in value.split(",") if item]

    def _parse_int_list(self, value: str) -> list[int]:
        return [int(item) for item in value.split(",") if item]

    def _handle_telemetry_line(self, line: str) -> None:
        if not line:
            return
        if line.startswith("TEL "):
            parsed: dict[str, object] = {}
            for item in line.split()[1:]:
                if "=" not in item:
                    continue
                key, raw = item.split("=", 1)
                if key == "ultra_cm":
                    parsed[key] = float(raw) if raw != "nan" else float("nan")
                elif key in {"rpm", "target"}:
                    parsed[key] = self._parse_float_list(raw)
                elif key in {"pwm", "count"}:
                    parsed[key] = self._parse_int_list(raw)
            self.telemetry = parsed
        elif line.startswith("ACK "):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    sequence = int(parts[1])
                except ValueError:
                    return
                if self.pending_velocity_seq == sequence:
                    self.last_acked_velocity = self.pending_velocity
                    self.last_ack_time = time.monotonic()
                    self.pending_velocity_seq = None
        elif line.startswith(("INFO", "FAULT", "WARN", "OK")):
            if line in {"OK VEL", "OK STOP"}:
                return
            print(f"\nESP32 {line}", flush=True)

    def poll(self) -> None:
        while True:
            try:
                waiting = getattr(self.esp32, "in_waiting", 0)
                if waiting <= 0:
                    break
                line = self.esp32.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    break
                self._handle_telemetry_line(line)
            except Exception:
                break

    def _rising_edge(self, controller_buttons: dict[str, int], name: str) -> bool:
        return bool(controller_buttons.get(name, 0)) and not bool(self.last_buttons.get(name, 0))

    def _emergency_stop_pressed(self, controller_buttons: dict[str, int]) -> bool:
        if self._rising_edge(controller_buttons, "guide"):
            return True
        back_pressed = bool(controller_buttons.get("back", 0))
        start_pressed = bool(controller_buttons.get("start", 0))
        if back_pressed and start_pressed:
            return self._rising_edge(controller_buttons, "back") or self._rising_edge(
                controller_buttons, "start"
            )
        return False

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.drive_deadzone else value

    def stop_drive(self) -> None:
        self._send_velocity(0.0, 0.0, 0.0, force=True)

    def _send_velocity_frame(self, sequence: int, velocity: tuple[float, float, float], now: float) -> None:
        strafe, forward, rotate = velocity
        self._write_esp32_line(f"VEL {sequence} {strafe:+.3f} {forward:+.3f} {rotate:+.3f}")
        self.pending_velocity_seq = sequence
        self.pending_velocity = velocity
        self.last_sent_vel = velocity
        self.last_velocity_send_time = now
        self.last_stop_sent = velocity == (0.0, 0.0, 0.0)

    def _send_velocity(self, strafe: float, forward: float, rotate: float, force: bool = False) -> None:
        next_vel = (strafe, forward, rotate)
        now = time.monotonic()
        changed = any(
            abs(current - previous) >= self.VELOCITY_CHANGE_EPSILON
            for current, previous in zip(next_vel, self.last_sent_vel)
        )

        if force or changed:
            sequence = self.next_velocity_seq
            self.next_velocity_seq += 1
            self._send_velocity_frame(sequence, next_vel, now)
            return

        if self.pending_velocity_seq is not None:
            if (now - self.last_velocity_send_time) < self.VELOCITY_RESEND_INTERVAL_S:
                return
            self._send_velocity_frame(self.pending_velocity_seq, self.pending_velocity, now)
            return

        interval = self.STOP_RESEND_INTERVAL_S if next_vel == (0.0, 0.0, 0.0) else self.VELOCITY_RESEND_INTERVAL_S
        if next_vel == self.last_acked_velocity and (now - self.last_ack_time) < interval:
            return

        sequence = self.next_velocity_seq
        self.next_velocity_seq += 1
        self._send_velocity_frame(sequence, next_vel, now)

    def _telemetry_summary(self) -> str:
        if not self.telemetry:
            return "tel=(waiting)"
        ultra = self.telemetry.get("ultra_cm", float("nan"))
        ultra_text = "nan" if ultra != ultra else f"{float(ultra):.1f}"
        rpm = self.telemetry.get("rpm", [])
        target = self.telemetry.get("target", [])
        if isinstance(rpm, list) and len(rpm) == 4 and isinstance(target, list) and len(target) == 4:
            rpm_text = format_wheel_values([round(float(v), 1) for v in rpm], precision=1)
            target_text = format_wheel_values([round(float(v), 1) for v in target], precision=1)
            return (
                "tel=("
                f"ultra={ultra_text} "
                f"rpm={rpm_text} "
                f"target={target_text}"
                ")"
            )
        return f"tel=(ultra={ultra_text})"

    def select_pair(self, pair_number: int) -> None:
        if not self.servo_enabled:
            return
        for index, (current_pair_number, _, _) in enumerate(self.valid_pairs):
            if current_pair_number == pair_number:
                self.selected_pair_index = index
                print(f"\nSelected {self._pair_label()}")
                return
        print(f"\nPair {pair_number} is not available on this controller.")

    def select_next_pair(self, step: int) -> None:
        if not self.servo_enabled:
            return
        self.selected_pair_index = (self.selected_pair_index + step) % len(self.valid_pairs)
        print(f"\nSelected {self._pair_label()}")

    def adjust_pair_step(self, delta_degrees: float) -> None:
        if not self.servo_enabled:
            return
        self.pair_step_deg = clamp_range(self.pair_step_deg + delta_degrees, 0.24, 3600.0)
        print(f"\nPair step set to {self.pair_step_deg:.1f}deg")

    def toggle_pair_control_mode(self) -> None:
        if not self.servo_enabled:
            return
        if self.pair_control_mode == "step":
            self.pair_control_mode = "motor"
        else:
            if not (self.fixed_step_ready or self.distance_step_ready):
                print("\nStep mode unavailable: live position read is not ready for pairs 1-3.")
                return
            self.pair_control_mode = "step"
        self.stop_selected_pair()
        print(f"\nSelected pair mode set to {self.pair_control_mode}")

    def _current_pair(self) -> tuple[int, int, int]:
        if not self.servo_enabled or not self.valid_pairs:
            return (0, 0, 0)
        return self.valid_pairs[self.selected_pair_index]

    def _pair_label(self) -> str:
        if not self.servo_enabled:
            return "pair=disabled"
        pair_number, primary, opposite = self._current_pair()
        return f"pair={pair_number} servos=({primary},{opposite})"

    def _is_pair4_selected(self) -> bool:
        if not self.servo_enabled or self.servo_mod is None:
            return False
        _, primary, opposite = self._current_pair()
        return (
            primary == self.servo_mod.PAIR4_PRIMARY
            and opposite == self.servo_mod.PAIR4_OPPOSITE
        )

    def _ensure_pair4_servo_mode(self) -> None:
        if not self.servo_enabled or self.servo_mod is None or self.servo is None:
            return
        self.servo.set_servo_mode(self.servo_mod.PAIR4_PRIMARY)
        self.servo.set_servo_mode(self.servo_mod.PAIR4_OPPOSITE)
        time.sleep(0.01)

    def _ensure_pair_servo_mode(self, primary: int, opposite: int) -> None:
        if not self.servo_enabled or self.servo is None:
            return
        self.servo.set_servo_mode(primary)
        self.servo.set_servo_mode(opposite)
        time.sleep(0.01)

    def _extract_position_value(self, value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(clamp_range(int(round(value)), 0, 1000))
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, bool):
                    continue
                if isinstance(item, (int, float)):
                    return int(clamp_range(int(round(item)), 0, 1000))
        return None

    def _extract_distance_value(self, value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(round(value))
        if isinstance(value, (bytes, bytearray)) and len(value) >= 4:
            return int.from_bytes(value[:4], byteorder="little", signed=True)
        if isinstance(value, (list, tuple)):
            ints = [item for item in value if isinstance(item, int) and not isinstance(item, bool)]
            if len(ints) >= 4 and all(0 <= item <= 255 for item in ints[:4]):
                return int.from_bytes(bytes(ints[:4]), byteorder="little", signed=True)
            for item in value:
                if isinstance(item, bool):
                    continue
                if isinstance(item, int):
                    return item
                if isinstance(item, float):
                    return int(round(item))
        return None

    def _read_servo_position(self, servo_id: int) -> int | None:
        if not self.servo_enabled or self.servo is None:
            return None
        for method_name in (
            "pos_read",
            "read_pos",
            "read_position",
            "position_read",
            "get_position",
        ):
            method = getattr(self.servo, method_name, None)
            if method is None:
                continue
            try:
                value = method(servo_id)
            except Exception:
                continue
            position = self._extract_position_value(value)
            if position is not None:
                return position
        return None

    def _read_servo_distance(self, servo_id: int) -> int | None:
        if not self.servo_enabled or self.servo is None:
            return None
        for method_name in (
            "dis_read",
            "read_dis",
            "read_distance",
            "distance_read",
            "get_distance",
        ):
            method = getattr(self.servo, method_name, None)
            if method is None:
                continue
            try:
                value = method(servo_id)
            except Exception:
                continue
            distance = self._extract_distance_value(value)
            if distance is not None:
                return distance
        return None

    def _default_center_position(self) -> int:
        if self.servo_mod is None:
            return 500
        degree_to_pos = getattr(self.servo_mod, "degree_to_pos", None)
        if callable(degree_to_pos):
            try:
                return int(clamp_range(int(round(degree_to_pos(120.0))), 0, 1000))
            except Exception:
                pass
        return 500

    def _get_live_servo_target(self, servo_id: int) -> int | None:
        current_position = self._read_servo_position(servo_id)
        if current_position is not None:
            self.pair_targets[servo_id] = current_position
            return current_position
        return None

    def _step_to_position_units(self) -> int:
        return max(1, int(round(self.pair_step_deg * 1000.0 / 240.0)))

    def _step_to_distance_units(self) -> int:
        return max(1, int(round(abs(self.pair_step_deg) * 4096.0 / 360.0)))

    def _check_fixed_step_support(self) -> bool:
        if not self.servo_enabled or self.servo_mod is None:
            return False
        ready = True
        checked_any = False
        for pair_number, primary, opposite in self.valid_pairs:
            if (
                primary == self.servo_mod.PAIR4_PRIMARY
                and opposite == self.servo_mod.PAIR4_OPPOSITE
            ):
                continue
            checked_any = True
            primary_pos = self._read_servo_position(primary)
            opposite_pos = self._read_servo_position(opposite)
            if primary_pos is None or opposite_pos is None:
                print(
                    f"Fixed-step self-check: pair {pair_number} missing live position read."
                )
                ready = False
                continue
            self.pair_targets[primary] = primary_pos
            self.pair_targets[opposite] = opposite_pos
        return ready or not checked_any

    def _check_distance_step_support(self) -> bool:
        if not self.servo_enabled or self.servo_mod is None:
            return False
        ready = True
        checked_any = False
        for _, primary, opposite in self.valid_pairs:
            if (
                primary == self.servo_mod.PAIR4_PRIMARY
                and opposite == self.servo_mod.PAIR4_OPPOSITE
            ):
                continue
            checked_any = True
            if self._read_servo_distance(primary) is None or self._read_servo_distance(opposite) is None:
                ready = False
        return ready and checked_any

    def move_pair4_mode(self, degrees: float) -> None:
        if not self.servo_enabled or self.servo_mod is None or self.servo is None:
            return
        if not self.pair4_available:
            return
        pos_primary = self.servo_mod.degree_to_pos(degrees)
        pos_opposite = self.servo_mod.mirror_pos(pos_primary)
        self._ensure_pair4_servo_mode()
        self.servo.move_time(self.servo_mod.PAIR4_PRIMARY, pos_primary, self.pair4_time_ms)
        self.servo.move_time(self.servo_mod.PAIR4_OPPOSITE, pos_opposite, self.pair4_time_ms)

    def move_selected_pair(self, direction: int) -> None:
        if not self.servo_enabled or self.servo is None:
            return
        if self._is_pair4_selected():
            return
        if self.distance_step_ready:
            self.start_distance_step(direction)
            return
        _, primary, opposite = self._current_pair()
        self._ensure_pair_servo_mode(primary, opposite)
        step_units = self._step_to_position_units()
        current_primary = self._get_live_servo_target(primary)
        current_opposite = self._get_live_servo_target(opposite)
        if current_primary is None or current_opposite is None:
            print(f"\n{self._pair_label()} live position read failed; skipping fixed-distance step.")
            return
        target_primary = int(clamp_range(current_primary + direction * step_units, 0, 1000))
        target_opposite = int(clamp_range(current_opposite - direction * step_units, 0, 1000))
        self.servo.move_time(primary, target_primary, self.pair_step_time_ms)
        self.servo.move_time(opposite, target_opposite, self.pair_step_time_ms)
        self.pair_targets[primary] = target_primary
        self.pair_targets[opposite] = target_opposite
        self.last_servo_motion = (self.selected_pair_index, direction)

    def start_distance_step(self, direction: int) -> None:
        if not self.servo_enabled or self.servo is None:
            return
        if self._is_pair4_selected():
            return
        if self.active_step_motion is not None:
            return
        _, primary, opposite = self._current_pair()
        primary_dist = self._read_servo_distance(primary)
        opposite_dist = self._read_servo_distance(opposite)
        if primary_dist is None or opposite_dist is None:
            return
        speed = max(1, min(1000, self.servo_speed))
        self.servo.set_motor_mode(primary, direction * speed)
        self.servo.set_motor_mode(opposite, -direction * speed)
        self.last_servo_motion = (self.selected_pair_index, direction * speed)
        self.active_step_motion = {
            "pair_index": self.selected_pair_index,
            "primary": primary,
            "opposite": opposite,
            "primary_start": primary_dist,
            "opposite_start": opposite_dist,
            "target_delta": self._step_to_distance_units(),
            "direction": direction,
            "primary_done": False,
            "opposite_done": False,
        }

    def update_active_step_motion(self) -> str | None:
        if not self.servo_enabled or self.servo is None:
            return None
        if self.active_step_motion is None:
            return None
        motion = self.active_step_motion
        primary = int(motion["primary"])
        opposite = int(motion["opposite"])
        primary_dist = self._read_servo_distance(primary)
        opposite_dist = self._read_servo_distance(opposite)
        if primary_dist is None or opposite_dist is None:
            self.stop_selected_pair()
            self.active_step_motion = None
            return "step_error=distance_read_failed"
        primary_done = bool(motion["primary_done"])
        opposite_done = bool(motion["opposite_done"])
        target_delta = int(motion["target_delta"])
        primary_start = int(motion["primary_start"])
        opposite_start = int(motion["opposite_start"])
        if not primary_done and abs(primary_dist - primary_start) >= target_delta:
            self.servo.set_motor_mode(primary, 0)
            motion["primary_done"] = True
            primary_done = True
        if not opposite_done and abs(opposite_dist - opposite_start) >= target_delta:
            self.servo.set_motor_mode(opposite, 0)
            motion["opposite_done"] = True
            opposite_done = True
        if primary_done and opposite_done:
            self.last_servo_motion = (self.selected_pair_index, 0)
            self.active_step_motion = None
            return "step_done"
        return (
            f"step_running="
            f"({abs(primary_dist - primary_start)}/{target_delta},"
            f"{abs(opposite_dist - opposite_start)}/{target_delta})"
        )

    def drive_selected_pair_motor(self, direction: int) -> None:
        if not self.servo_enabled or self.servo is None:
            return
        if self._is_pair4_selected():
            return
        _, primary, opposite = self._current_pair()
        speed = max(-1000, min(1000, direction * self.servo_speed))
        motion_key = (self.selected_pair_index, speed)
        if self.last_servo_motion == motion_key:
            return
        self.servo.set_motor_mode(primary, speed)
        self.servo.set_motor_mode(opposite, -speed)
        self.last_servo_motion = motion_key

    def _stop_servo_move(self, servo_id: int) -> bool:
        if not self.servo_enabled or self.servo is None:
            return False
        for method_name in ("move_stop", "stop_move", "servo_move_stop"):
            method = getattr(self.servo, method_name, None)
            if method is None:
                continue
            try:
                method(servo_id)
                return True
            except Exception:
                continue
        return False

    def stop_selected_pair(self) -> None:
        if not self.servo_enabled or self.servo is None:
            return
        _, primary, opposite = self._current_pair()
        if self.active_step_motion is not None:
            try:
                active_primary = int(self.active_step_motion["primary"])
                active_opposite = int(self.active_step_motion["opposite"])
                self.servo.set_motor_mode(active_primary, 0)
                self.servo.set_motor_mode(active_opposite, 0)
            except Exception:
                pass
            self.active_step_motion = None
        stopped = self._stop_servo_move(primary) | self._stop_servo_move(opposite)
        if not stopped and not self._is_pair4_selected():
            self.servo.set_motor_mode(primary, 0)
            self.servo.set_motor_mode(opposite, 0)
        self.last_servo_motion = (self.selected_pair_index, 0)

    def stop_all_pairs_1_to_3(self) -> None:
        if not self.servo_enabled or self.servo is None or self.servo_mod is None:
            return
        self.active_step_motion = None
        for _, primary, opposite in self.valid_pairs:
            if (
                primary == self.servo_mod.PAIR4_PRIMARY
                and opposite == self.servo_mod.PAIR4_OPPOSITE
            ):
                continue
            try:
                if not self._stop_servo_move(primary):
                    self.servo.set_motor_mode(primary, 0)
                if not self._stop_servo_move(opposite):
                    self.servo.set_motor_mode(opposite, 0)
            except Exception:
                pass
        self.last_servo_motion = None

    def emergency_stop(self) -> None:
        self.stop_drive()
        self.stop_all_pairs_1_to_3()

    def drive_omni(self, controller_axes: dict[str, float]) -> tuple[float, float, float, str]:
        strafe = self._apply_deadzone(controller_axes.get("leftx", 0.0)) * self.drive_scale
        forward = -self._apply_deadzone(controller_axes.get("lefty", 0.0)) * self.drive_scale
        rotate = self._apply_deadzone(controller_axes.get("rightx", 0.0)) * self.drive_scale
        if abs(strafe) < self.command_deadzone and abs(forward) < self.command_deadzone and abs(rotate) < self.command_deadzone:
            self.stop_drive()
            return 0.0, 0.0, 0.0, "stop"
        self._send_velocity(strafe, forward, rotate)
        return strafe, forward, rotate, "omni"

    def handle_packet(
        self,
        controller_axes: dict[str, float],
        controller_buttons: dict[str, int],
        imu_mag_x_ut: float | None = None,
    ) -> str:
        _ = imu_mag_x_ut
        if self._emergency_stop_pressed(controller_buttons):
            print("\nEmergency stop.")
            self.emergency_stop()
            self.last_buttons = dict(controller_buttons)
            return (
                f"{self._pair_label()} pair_mode={self.pair_control_mode} "
                "drive=(stopped) servo=(stopped) "
                f"{self._telemetry_summary()}"
            )

        strafe, forward, rotate, drive_mode = self.drive_omni(controller_axes)
        step_progress = self.update_active_step_motion() if self.servo_enabled else None

        if self._rising_edge(controller_buttons, "back"):
            print("\nDrive stop.")
            self.stop_drive()
            strafe = 0.0
            forward = 0.0
            rotate = 0.0
            drive_mode = "stop"

        if self.servo_enabled:
            if self._rising_edge(controller_buttons, "a"):
                self.select_pair(1)
            if self._rising_edge(controller_buttons, "b"):
                self.select_pair(2)
            if self._rising_edge(controller_buttons, "x"):
                self.select_pair(3)
            if self._rising_edge(controller_buttons, "y"):
                self.toggle_pair_control_mode()
            if self._rising_edge(controller_buttons, "leftstick"):
                self.select_next_pair(-1)
            if self._rising_edge(controller_buttons, "rightstick"):
                self.select_next_pair(1)
            if self._rising_edge(controller_buttons, "dpad_up"):
                self.adjust_pair_step(2.0)
            if self._rising_edge(controller_buttons, "dpad_down"):
                self.adjust_pair_step(-2.0)
            if self._rising_edge(controller_buttons, "dpad_left") and self.pair4_available:
                self.move_pair4_mode(self.pair4_mode1_deg)
            if self._rising_edge(controller_buttons, "dpad_right") and self.pair4_available:
                self.move_pair4_mode(self.pair4_mode2_deg)
            if self._rising_edge(controller_buttons, "start"):
                print(f"\nStopping {self._pair_label()}")
                self.stop_selected_pair()

            if self._is_pair4_selected():
                servo_state = "pair4 presets"
            elif self.pair_control_mode == "step":
                if self._rising_edge(controller_buttons, "leftshoulder"):
                    self.move_selected_pair(-1)
                    servo_state = "step=-1"
                elif self._rising_edge(controller_buttons, "rightshoulder"):
                    self.move_selected_pair(1)
                    servo_state = "step=+1"
                elif step_progress is not None:
                    servo_state = step_progress
                else:
                    servo_state = "idle"
            elif controller_buttons.get("leftshoulder", 0):
                self.drive_selected_pair_motor(-1)
                servo_state = "motor=-1"
            elif controller_buttons.get("rightshoulder", 0):
                self.drive_selected_pair_motor(1)
                servo_state = "motor=+1"
            else:
                self.stop_selected_pair()
                servo_state = "idle"
        else:
            servo_state = "disabled"

        self.last_buttons = dict(controller_buttons)
        return (
            f"{self._pair_label()} pair_mode={self.pair_control_mode} "
            f"pair_step={self.pair_step_deg:.1f}deg/{self.pair_step_time_ms}ms "
            f"drive=(mode={drive_mode} x={strafe:+.2f} y={forward:+.2f} rot={rotate:+.2f}) "
            f"servo=({servo_state}) {self._telemetry_summary()}"
        )

    def on_timeout(self) -> None:
        self.emergency_stop()

    def close(self) -> None:
        try:
            self.emergency_stop()
        finally:
            try:
                self.esp32.close()
            finally:
                if self.servo is not None:
                    self.servo.close()


def _fake_disabled(value: object) -> bool:
    return str(value or "").strip().lower() in {
        "",
        "none",
        "off",
        "disable",
        "disabled",
        "false",
        "0",
        "null",
    }


def _fake_port(configured: object, fallback: str) -> str:
    text = str(configured or "").strip()
    if not text or text.lower() == "auto":
        return fallback
    return text


def _fake_missing_servo_ids(text: str) -> set[int]:
    missing_ids: set[int] = set()
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            missing_ids.add(int(item))
        except ValueError:
            continue
    return missing_ids


def _print_fake_servo_pair_check(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    missing_ids = _fake_missing_servo_ids(args.fake_missing_servos)
    servo_pairs = [(1, 5), (2, 6), (3, 7), (4, 8)]
    valid_pairs: list[tuple[int, int, int]] = []

    print("\n" + "=" * 60)
    print("VERIFYING SERVO PAIRS")
    print("=" * 60)
    for pair_idx, (primary, opposite) in enumerate(servo_pairs, 1):
        primary_ok = primary not in missing_ids
        opposite_ok = opposite not in missing_ids
        if primary_ok and opposite_ok:
            print(f"  OK Pair {pair_idx}: Servo {primary} & {opposite}")
            valid_pairs.append((pair_idx, primary, opposite))
        else:
            if not primary_ok:
                print(f"  MISS Pair {pair_idx}: Servo {primary}")
            if not opposite_ok:
                print(f"  MISS Pair {pair_idx}: Servo {opposite}")

    if not valid_pairs:
        print("\nNo valid pairs found.")
    else:
        print(f"\nFound {len(valid_pairs)} valid pair(s).")

    return valid_pairs


def _fake_controller_snapshot(
    elapsed_s: float,
    controller_name: str,
) -> tuple[dict[str, float], dict[str, int], str]:
    forward = clamp_range(0.42 + 0.18 * math.sin(elapsed_s * 0.75), -1.0, 1.0)
    strafe = clamp_range(0.22 * math.sin(elapsed_s * 0.55), -1.0, 1.0)
    rotate = clamp_range(0.18 * math.sin(elapsed_s * 0.95 + 0.4), -1.0, 1.0)
    controller_axes = {
        "leftx": strafe,
        "lefty": -forward,
        "rightx": rotate,
        "righty": 0.0,
        "triggerleft": 0.0,
        "triggerright": 0.0,
    }

    controller_buttons = {name: 0 for name in BUTTON_ORDER}
    scripted_button = {
        1: "a",
        2: "rightshoulder",
        3: "y",
        4: "dpad_right",
        5: "leftstick",
        6: "rightstick",
        7: "back",
    }.get(int(elapsed_s / 1.4) % 10)
    if scripted_button is not None:
        controller_buttons[scripted_button] = 1
    return controller_axes, controller_buttons, controller_name


def _fake_ultra_values(elapsed_s: float) -> list[float | None]:
    return [
        31.5 + 3.2 * math.sin(elapsed_s * 0.70),
        44.0 + 2.5 * math.sin(elapsed_s * 0.45 + 1.5),
        27.5 + 4.0 * math.sin(elapsed_s * 0.62 + 2.1),
        None if int(elapsed_s / 2.8) % 4 == 0 else 58.0 + 5.0 * math.sin(elapsed_s * 0.35),
    ]


def _fake_ultra_serial_status(args: argparse.Namespace, elapsed_s: float) -> str:
    if not args.ultra_enabled or _fake_disabled(args.ultra_port):
        return "ultra_serial=(disabled)"

    parts: list[str] = []
    for value in _fake_ultra_values(elapsed_s):
        if value is None:
            parts.append("NR")
        else:
            parts.append(f"{value:.1f}")
    return f"ultra_serial=({','.join(parts)}cm age=0.02s)"


def _fake_imu_status(args: argparse.Namespace, elapsed_s: float) -> str:
    if not args.imu_enabled or _fake_disabled(args.imu_port):
        return "imu=(disabled)"
    mag_x = 40.25 + 1.4 * math.sin(elapsed_s * 0.38) + min(elapsed_s * 0.08, 1.8)
    return f"imu=(x={mag_x:+.3f}uT age=0.02s)"


def _fake_wheel_targets(
    args: argparse.Namespace,
    strafe: float,
    forward: float,
    rotate: float,
) -> list[float]:
    raw_values = [
        forward + strafe + rotate,
        forward - strafe - rotate,
        forward + strafe - rotate,
        forward - strafe + rotate,
    ]
    return [
        clamp_range(value, -1.0, 1.0) * float(args.max_wheel_rpm)
        for value in raw_values
    ]


def _fake_hardware_telemetry(
    args: argparse.Namespace,
    elapsed_s: float,
    target_rpm: list[float],
) -> str:
    if args.hardware == "amd-yes-v3" and _fake_disabled(args.esp32_port):
        return "tel=(disabled)"

    ultra_values = [value for value in _fake_ultra_values(elapsed_s) if value is not None]
    ultra = min(ultra_values) if ultra_values else float("nan")
    rpm = [
        target * (0.88 + 0.06 * math.sin(elapsed_s * 1.3 + index))
        for index, target in enumerate(target_rpm)
    ]
    rpm_text = ",".join(f"{value:+.0f}" for value in rpm)

    if args.hardware == "amd-yes-v3":
        return f"tel=(ultra={ultra:.1f} rpm={rpm_text})"

    return f"tel=(ultra={ultra:.1f} rpm={rpm_text})"


def _fake_servo_state(controller_buttons: dict[str, int]) -> str:
    if controller_buttons.get("leftshoulder", 0):
        return "motor=-1"
    if controller_buttons.get("rightshoulder", 0):
        return "motor=+1"
    if controller_buttons.get("dpad_left", 0) or controller_buttons.get("dpad_right", 0):
        return "pair4"
    if controller_buttons.get("back", 0):
        return "stop"
    return "idle"


def _fake_status(
    args: argparse.Namespace,
    elapsed_s: float,
    controller_axes: dict[str, float],
    controller_buttons: dict[str, int],
) -> str:
    forward = clamp_range(-controller_axes.get("lefty", 0.0), -1.0, 1.0)
    turn = clamp_range(controller_axes.get("leftx", 0.0), -1.0, 1.0)
    rotate = clamp_range(controller_axes.get("rightx", 0.0), -1.0, 1.0)

    if args.hardware == "generic":
        left, right = mix_drive(forward, turn, args.max_output)
        return f"left={left:+.3f} right={right:+.3f}"

    target_rpm = _fake_wheel_targets(args, turn, forward, rotate)
    telemetry = _fake_hardware_telemetry(args, elapsed_s, target_rpm)
    pair_index = int(elapsed_s / 5.0) % 4
    pair_number = pair_index + 1
    servo_state = "disabled" if _fake_disabled(args.servo_port) else _fake_servo_state(controller_buttons)
    drive_mode = "auto-left" if controller_buttons.get("rightstick", 0) else "proportional"

    return (
        f"pair={pair_number} mode={drive_mode} "
        f"drive=(x={turn:+.2f} y={forward:+.2f} r={rotate:+.2f}) "
        f"servo={servo_state} {telemetry}"
    )


def _print_fake_startup(args: argparse.Namespace) -> None:
    print(
        "FAKE MODE: report-capture simulation only; no UDP socket, serial ports, "
        "motors, servos, IMU, or ultrasound hardware are opened."
    )
    servo_enabled = not _fake_disabled(args.servo_port)
    valid_servo_pairs: list[tuple[int, int, int]] = []
    if servo_enabled and args.hardware in {"esp32-omni", "amd-yes-v3"}:
        valid_servo_pairs = _print_fake_servo_pair_check(args)

    if args.hardware == "esp32-omni":
        esp32_port = _fake_port(args.esp32_port, "/dev/ttyACM0")
        print(
            "ESP32-S3 omni drive ready: "
            f"esp32_port={esp32_port} servo_port={args.servo_port} "
            f"servo_enabled={servo_enabled} selected_pair={1 if valid_servo_pairs else 'n/a'}"
        )
        print(
            "Mapping: left stick = proportional strafe/forward, right stick X = proportional rotation, "
            "A/B/X = select pair 1/2/3, Y = toggle pair mode, "
            "left/right stick click = cycle pairs, LB/RB = move selected pair, "
            "D-pad left/right = pair4 presets, D-pad up/down = pair step size, "
            "Start = stop selected pair, Back = stop drive, Guide = emergency stop all."
        )
        if valid_servo_pairs:
            print("Fixed-distance stepping ready: live position read OK for pairs 1-3.")
            print("Extended step ready: live distance read OK for pairs 1-3.")
            print(
                "Pair 4 position presets: "
                f"{args.pair4_mode1_deg:.1f}deg / {args.pair4_mode2_deg:.1f}deg "
                f"(step={args.pair_step_deg:.1f}deg/{args.pair_step_time_ms}ms, motor_speed={args.servo_speed})"
            )
        elif servo_enabled:
            print("Servo pair check found no valid pairs; continuing fake output for capture.")
    elif args.hardware == "amd-yes-v3":
        if _fake_disabled(args.esp32_port):
            telemetry_text = "telemetry=disabled"
        else:
            telemetry_text = f"telemetry_port={_fake_port(args.esp32_port, '/dev/ttyACM0')}"
        print(
            "AMD_YES v3 ready: "
            f"motor_driver=original_v2_driver_board {telemetry_text}"
        )
        if valid_servo_pairs:
            print("Fixed-distance stepping ready: live position read OK for pairs 1-3.")
            print("Extended step ready: live distance read OK for pairs 1-3.")
        elif servo_enabled:
            print("Servo pair check found no valid pairs; continuing fake output for capture.")
    else:
        print("Generic stdout drive ready: no hardware device opened")

    print(f"Listening for UDP teleop on {args.bind_host}:{args.port}")
    print(f"Hardware mode: {args.hardware}")
    if args.imu_enabled and not _fake_disabled(args.imu_port):
        backend = "ybimulib" if args.imu_backend == "auto" else args.imu_backend
        print(
            f"Reading IMU serial on {_fake_port(args.imu_port, '/dev/ttyUSB0')} "
            f"@ {args.imu_baud} backend={backend} "
            f"(timeout={max(0.05, args.imu_timeout):.2f}s)"
        )
    else:
        print("IMU telemetry: disabled")

    if args.ultra_enabled and not _fake_disabled(args.ultra_port):
        print(
            f"Reading ultrasound serial on {_fake_port(args.ultra_port, '/dev/ttyCH343USB0')} "
            f"@ {args.ultra_baud} (timeout={max(0.05, args.ultra_timeout):.2f}s)"
        )
    else:
        print("Ultrasound telemetry: disabled")
    print("Press Ctrl+C to stop.")


def run_fake_receiver(args: argparse.Namespace) -> int:
    status_print_interval_s = 0.15
    idle_sleep_s = 0.005
    _print_fake_startup(args)

    start_time = time.monotonic()
    last_status_print_time = 0.0
    try:
        while True:
            now = time.monotonic()
            elapsed_s = now - start_time
            if args.fake_duration > 0.0 and elapsed_s >= args.fake_duration:
                break
            if (now - last_status_print_time) >= status_print_interval_s:
                controller_axes, controller_buttons, controller_name = _fake_controller_snapshot(
                    elapsed_s,
                    args.fake_controller_name,
                )
                status = _fake_status(args, elapsed_s, controller_axes, controller_buttons)
                status = (
                    f"{status} {_fake_imu_status(args, elapsed_s)} "
                    f"{_fake_ultra_serial_status(args, elapsed_s)}"
                )
                source_port = 51732 + (int(elapsed_s) % 100)
                print(
                    "\r\033[2K" + format_status_line(
                        (args.fake_source_ip, source_port),
                        controller_name,
                        status,
                        controller_buttons,
                    ),
                    end="",
                    flush=True,
                )
                last_status_print_time = now
            time.sleep(idle_sleep_s)
    except KeyboardInterrupt:
        print("\nStopping receiver and commanding safe stop.")
        print("Fake outputs stopped.")
        return 0

    print("\nStopping receiver and commanding safe stop.")
    print("Fake outputs stopped.")
    return 0


def main() -> int:
    status_print_interval_s = 0.15
    idle_sleep_s = 0.005
    args = parse_args()

    if args.fake:
        return run_fake_receiver(args)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, args.port))
    sock.setblocking(False)
    imu_reader: ImuSerialReader | None = None
    if args.imu_enabled:
        imu_reader = ImuSerialReader(
            port=args.imu_port,
            baud=args.imu_baud,
            timeout_s=args.imu_timeout,
            backend=args.imu_backend,
        )
    ultra_reader: UltraSerialReader | None = None
    if args.ultra_enabled:
        ultra_reader = UltraSerialReader(
            port=args.ultra_port,
            baud=args.ultra_baud,
            timeout_s=args.ultra_timeout,
        )

    if args.hardware == "esp32-omni":
        hardware: GenericDriveController | Esp32S3OmniController | AmdYesTelemetryV3Controller = Esp32S3OmniController(args)
    elif args.hardware == "amd-yes-v3":
        hardware = AmdYesTelemetryV3Controller(args)
    else:
        hardware = GenericDriveController()

    last_packet_time = 0.0
    last_status_print_time = 0.0
    link_active = False
    stopped_due_to_timeout = False

    print(f"Listening for UDP teleop on {args.bind_host}:{args.port}")
    print(f"Hardware mode: {args.hardware}")
    if args.imu_enabled and imu_reader is not None:
        print(f"{imu_reader.source_text()} (timeout={max(0.05, args.imu_timeout):.2f}s)")
    elif not args.imu_enabled:
        print("IMU telemetry: disabled")
    if args.ultra_enabled and ultra_reader is not None:
        print(f"{ultra_reader.source_text()} (timeout={max(0.05, args.ultra_timeout):.2f}s)")
    elif not args.ultra_enabled:
        print("Ultrasound telemetry: disabled")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            now = time.monotonic()
            hardware.poll()
            imu_mag_x_ut_for_control: float | None = None
            if imu_reader is not None:
                imu_reader.poll(now)
                imu_mag_x_ut_for_control = imu_reader.current_mag_x_ut(now, fresh_only=True)
            if ultra_reader is not None:
                ultra_reader.poll(now)
            try:
                latest_packet: tuple[bytes, tuple[str, int]] | None = None
                while True:
                    latest_packet = sock.recvfrom(4096)
            except BlockingIOError:
                if latest_packet is None:
                    if link_active and (now - last_packet_time) > args.timeout:
                        if not stopped_due_to_timeout:
                            hardware.on_timeout()
                            print("\nLink timeout: stopping outputs.", flush=True)
                            stopped_due_to_timeout = True
                        link_active = False
                    time.sleep(idle_sleep_s)
                    continue

                data, addr = latest_packet
                message = data.decode("utf-8").strip()
                (
                    forward,
                    turn,
                    axes,
                    buttons,
                    hats,
                    controller_axes,
                    controller_buttons,
                    controller_name,
                ) = parse_packet(message)

                last_packet_time = now
                link_active = True
                stopped_due_to_timeout = False

                if args.hardware in {"esp32-omni", "amd-yes-v3"}:
                    status = hardware.handle_packet(
                        controller_axes,
                        controller_buttons,
                        imu_mag_x_ut=imu_mag_x_ut_for_control,
                    )
                    imu_status = "imu=(disabled)" if not args.imu_enabled else (
                        imu_reader.status_text(now) if imu_reader is not None else "imu=(waiting)"
                    )
                    ultra_status = "ultra_serial=(disabled)" if not args.ultra_enabled else (
                        ultra_reader.status_text(now)
                        if ultra_reader is not None
                        else "ultra_serial=(waiting)"
                    )
                    status = f"{status} {imu_status} {ultra_status}"
                    if (now - last_status_print_time) >= status_print_interval_s:
                        print(
                            "\r\033[2K" + format_status_line(
                                addr,
                                controller_name,
                                status,
                                controller_buttons,
                            ),
                            end="",
                            flush=True,
                        )
                        last_status_print_time = now
                else:
                    left, right = mix_drive(forward, turn, args.max_output)
                    hardware.send_motor_command(left, right)
                    print(f"\rleft={left:+.3f} right={right:+.3f}", end="", flush=True)
            except (ValueError, json.JSONDecodeError):
                print("\nIgnoring malformed packet.", flush=True)
    except KeyboardInterrupt:
        print("\nStopping receiver and commanding safe stop.")
        hardware.on_timeout()
        return 0
    finally:
        sock.close()
        if imu_reader is not None:
            imu_reader.close()
        if ultra_reader is not None:
            ultra_reader.close()
        hardware.close()


if __name__ == "__main__":
    raise SystemExit(main())

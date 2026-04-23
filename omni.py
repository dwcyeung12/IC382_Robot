#!/usr/bin/env python3
"""
Omni wheel 4-motor car control (Bus370) + ultrasonic safety stop.

Behavior:
- Keyboard controls: W/A/S/D, Q/E rotate, SPACE stop, X quit
- Reads HC-SR04 distance from Arduino Uno over USB serial
- If distance is below threshold (default 65 cm), forward motion is blocked/stopped

Motor mapping:
- 0 = front L
- 2 = front R
- 3 = rear R
- 1 = rear L
"""

import argparse
import select
import sys
import termios
import threading
import time
import tty

import serial
from serial.tools import list_ports

# Motor IDs
MOTOR_FL = 0
MOTOR_FR = 2
MOTOR_RR = 3
MOTOR_RL = 1

# Per-motor direction multipliers (use -1 if a motor spins opposite)
DIR_FL = 1
DIR_FR = -1
DIR_RR = -1
DIR_RL = 1

DEFAULT_MOTOR_PORT = "/dev/ttyUSB0"
DEFAULT_ULTRA_PORT = "/dev/ttyACM0"
DEFAULT_ULTRA_BAUD = 9600
DEFAULT_STOP_DISTANCE_CM = 65.0


class DistanceReader:
    """Continuously reads distance lines from Arduino serial output."""

    def __init__(self, port: str, baud: int, debug: bool = False):
        self.port = port
        self.baud = baud
        self.debug = debug
        self.ser = None
        self.lock = threading.Lock()
        self.latest_cm = None
        self.out_of_range = False
        self.running = False
        self.thread = None

    def start(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        # Wait for Uno auto-reset
        time.sleep(2)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass

    def get_status(self):
        with self.lock:
            return self.latest_cm, self.out_of_range

    def _loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                time.sleep(0.05)
                continue

            if not line:
                continue

            if line == "OUT_OF_RANGE":
                with self.lock:
                    self.out_of_range = True
                    self.latest_cm = None
                if self.debug:
                    print("[ULTRA] OUT_OF_RANGE")
                continue

            try:
                val = float(line)
                with self.lock:
                    self.latest_cm = val
                    self.out_of_range = False
                if self.debug:
                    print(f"[ULTRA] {val:.2f} cm")
            except ValueError:
                if self.debug:
                    print(f"[ULTRA] Invalid data: {line}")


def fmt_id(x: int) -> str:
    return f"{max(0, min(255, x)):03d}"


def fmt_pwm(x: int) -> str:
    x = max(500, min(2500, x))
    return f"{x:04d}"


def fmt_time_s(x: int) -> str:
    x = max(0, min(9999, x))
    return f"{x:04d}"


class Bus370:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.2, debug: bool = False):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.debug = debug
        self.ser = serial.Serial(port, baud, timeout=timeout, write_timeout=timeout)
        time.sleep(0.15)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def _tx(self, cmd: str):
        b = cmd.encode("ascii")
        if self.debug:
            print("[TX]", cmd)
        self.ser.write(b)
        self.ser.flush()

    def transact(self, cmd: str):
        self.ser.reset_input_buffer()
        self._tx(cmd)

    def set_pwm(self, motor_id: int, pwm: int, time_s: int = 0):
        cmd = f"#{fmt_id(motor_id)}P{fmt_pwm(pwm)}T{fmt_time_s(time_s)}!"
        self.transact(cmd)

    def stop_motor(self, motor_id: int, time_s: int = 1):
        self.set_pwm(motor_id, 1500, time_s)


def scan_usb_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No USB ports found!")
        return

    print("\n" + "=" * 60)
    print("AVAILABLE USB PORTS")
    print("=" * 60)
    for p in ports:
        vid = f"{p.vid:04X}" if p.vid is not None else "----"
        pid = f"{p.pid:04X}" if p.pid is not None else "----"
        ch340_marker = " [CH340 - Motor Controller]" if (p.vid == 0x1A86 and p.pid == 0x7523) else ""
        print(f"Port: {p.device}")
        print(f"  Description: {p.description}")
        print(f"  VID:PID: {vid}:{pid}{ch340_marker}")
        print()
    print("=" * 60 + "\n")


def build_pwm(fwd_pwm: int, rev_pwm: int, direction: int) -> int:
    if direction >= 0:
        return fwd_pwm
    return rev_pwm


def apply_drive(bus: Bus370, fwd_pwm: int, rev_pwm: int, fl: int, fr: int, rr: int, rl: int):
    bus.set_pwm(MOTOR_FL, build_pwm(fwd_pwm, rev_pwm, fl))
    bus.set_pwm(MOTOR_FR, build_pwm(fwd_pwm, rev_pwm, fr))
    bus.set_pwm(MOTOR_RR, build_pwm(fwd_pwm, rev_pwm, rr))
    bus.set_pwm(MOTOR_RL, build_pwm(fwd_pwm, rev_pwm, rl))


def stop_all(bus: Bus370):
    bus.stop_motor(MOTOR_FL)
    bus.stop_motor(MOTOR_FR)
    bus.stop_motor(MOTOR_RR)
    bus.stop_motor(MOTOR_RL)


def print_speed(fwd_pwm, rev_pwm):
    print(f"Speed: FWD={fwd_pwm} REV={rev_pwm}")


def read_char_with_timeout(timeout_s: float):
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    return sys.stdin.read(1)


def can_move_forward(distance_reader: DistanceReader, min_cm: float) -> bool:
    dist, out_of_range = distance_reader.get_status()
    if out_of_range:
        return True
    if dist is None:
        return True
    return dist >= min_cm


def print_forward_block(distance_reader: DistanceReader, min_cm: float):
    dist, out_of_range = distance_reader.get_status()
    if out_of_range or dist is None:
        print(f"FORWARD BLOCKED: obstacle detected (distance unknown), threshold={min_cm:.1f} cm")
    else:
        print(f"FORWARD BLOCKED: {dist:.2f} cm < {min_cm:.1f} cm")


def main():
    ap = argparse.ArgumentParser(description="Omni wheel 4-motor + ultrasonic forward safety stop")
    ap.add_argument("--motor-port", default=DEFAULT_MOTOR_PORT, help="Motor serial port")
    ap.add_argument("--motor-baud", type=int, default=115200, help="Motor controller baud rate")
    ap.add_argument("--ultra-port", default=DEFAULT_ULTRA_PORT, help="Arduino Uno serial port")
    ap.add_argument("--ultra-baud", type=int, default=DEFAULT_ULTRA_BAUD, help="Ultrasonic serial baud")
    ap.add_argument("--stop-distance-cm", type=float, default=DEFAULT_STOP_DISTANCE_CM, help="Forward stop threshold")
    ap.add_argument("--fwd-pwm", type=int, default=1700, help="Forward PWM")
    ap.add_argument("--rev-pwm", type=int, default=1300, help="Reverse PWM")
    ap.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = ap.parse_args()

    scan_usb_ports()
    print(f"Using motor port: {args.motor_port}")
    print(f"Using ultrasonic port: {args.ultra_port}")
    print(f"Forward stop threshold: {args.stop_distance_cm:.1f} cm")

    bus = Bus370(args.motor_port, baud=args.motor_baud, timeout=1.2, debug=args.debug)
    distance_reader = DistanceReader(args.ultra_port, args.ultra_baud, debug=args.debug)
    distance_reader.start()

    def print_controls():
        print("\n" + "=" * 60)
        print("OMNI CAR CONTROL + ULTRASONIC SAFETY")
        print("=" * 60)
        print("W = Forward (blocked if distance < threshold)")
        print("S = Reverse")
        print("A = Strafe Left")
        print("D = Strafe Right")
        print("Q = Rotate Left")
        print("E = Rotate Right")
        print("UP/DOWN = Adjust speed")
        print("SPACE = Stop")
        print("X = Quit")
        print("=" * 60 + "\n")

    print_controls()

    fwd_pwm = args.fwd_pwm
    rev_pwm = args.rev_pwm
    print_speed(fwd_pwm, rev_pwm)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    print("Waiting for input...")

    motion_state = "stop"

    try:
        stop_all(bus)

        while True:
            # If currently moving forward and obstacle gets too close, stop immediately.
            if motion_state == "forward" and not can_move_forward(distance_reader, args.stop_distance_cm):
                print_forward_block(distance_reader, args.stop_distance_cm)
                print("AUTO STOP")
                stop_all(bus)
                motion_state = "stop"

            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue

            ch = sys.stdin.read(1)

            if ch == "\x1b":  # Arrow key escape sequence prefix
                # Arrow keys arrive as an escape sequence: ESC [ <key>.
                # Peek briefly for the next bytes; standalone ESC is ignored.
                next_ch = read_char_with_timeout(0.03)
                if next_ch == "[":
                    arrow = read_char_with_timeout(0.03)
                    if arrow == "A":  # UP
                        # Increase forward command and mirror reverse command.
                        # Keeping them symmetric preserves the neutral midpoint around 1500.
                        fwd_pwm = min(2500, fwd_pwm + 50)
                        rev_pwm = max(500, rev_pwm - 50)
                        print_speed(fwd_pwm, rev_pwm)
                        continue
                    if arrow == "B":  # DOWN
                        # Decrease forward command and mirror reverse command.
                        fwd_pwm = max(500, fwd_pwm - 50)
                        rev_pwm = min(2500, rev_pwm + 50)
                        print_speed(fwd_pwm, rev_pwm)
                        continue
                    # Ignore LEFT/RIGHT and unknown CSI key sequences
                    continue
                # Ignore standalone ESC and any non-arrow ESC sequence.
                continue

            ch = ch.lower()

            if ch == "x":
                break

            if ch == " ":
                print("STOP")
                stop_all(bus)
                motion_state = "stop"
                continue

            if ch == "w":
                if not can_move_forward(distance_reader, args.stop_distance_cm):
                    print_forward_block(distance_reader, args.stop_distance_cm)
                    stop_all(bus)
                    motion_state = "stop"
                    continue

                print("FORWARD")
                apply_drive(bus, fwd_pwm, rev_pwm, DIR_FL, DIR_FR, DIR_RR, DIR_RL)
                motion_state = "forward"
                continue

            if ch == "s":
                print("REVERSE")
                apply_drive(bus, fwd_pwm, rev_pwm, -DIR_FL, -DIR_FR, -DIR_RR, -DIR_RL)
                motion_state = "reverse"
                continue

            if ch == "a":
                print("STRAFE LEFT")
                apply_drive(bus, fwd_pwm, rev_pwm, -DIR_FL, DIR_FR, -DIR_RR, DIR_RL)
                motion_state = "strafe_left"
                continue

            if ch == "d":
                print("STRAFE RIGHT")
                apply_drive(bus, fwd_pwm, rev_pwm, DIR_FL, -DIR_FR, DIR_RR, -DIR_RL)
                motion_state = "strafe_right"
                continue

            if ch == "q":
                print("ROTATE LEFT")
                apply_drive(bus, fwd_pwm, rev_pwm, -DIR_FL, DIR_FR, DIR_RR, -DIR_RL)
                motion_state = "rotate_left"
                continue

            if ch == "e":
                print("ROTATE RIGHT")
                apply_drive(bus, fwd_pwm, rev_pwm, DIR_FL, -DIR_FR, -DIR_RR, DIR_RL)
                motion_state = "rotate_right"
                continue

    finally:
        stop_all(bus)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        distance_reader.stop()
        bus.close()
        print("\nExit.")


if __name__ == "__main__":
    main()

# IC382_Robot

Python scripts for teleoperation, omni-wheel drive control, IMU/ultrasonic serial integration, and cube face detection.

## Repository contents

- `mac_teleop_sender.py`  
  Reads an Xbox-style controller with `pygame` and sends controller state over UDP.

- `jetson_udp_receiver_v5.py`  
  Receives UDP controller packets and drives hardware backends (`generic`, `esp32-omni`, `amd-yes-v3`) with optional IMU and ultrasonic telemetry.

- `omni.py`  
  Keyboard-based omni-wheel controller for a 4-motor platform with ultrasonic forward safety stop.

- `imu_Serial.py`  
  Reads IMU serial data and writes the latest magnetometer X value to a shared JSON file (default: `/tmp/imu_mag_latest.json`).

- `test016.py`  
  Vision script for cube/box detection and face-on estimation with optional distance overlay (OpenCV + YOLO-World).

## Requirements

- Python 3.10+ recommended
- For teleop/control scripts:
  - `pyserial`
  - `pygame`
- For vision script (`test016.py`):
  - `opencv-python`
  - `numpy`
  - `torch`
  - `ultralytics`

Example install:

```bash
python3 -m pip install pyserial pygame opencv-python numpy torch ultralytics
```

## Quick start

### 1) Send controller data (operator machine)

```bash
python3 mac_teleop_sender.py --jetson-ip <JETSON_IP> --port 5005
```

### 2) Receive and drive robot (robot/Jetson machine)

```bash
python3 jetson_udp_receiver_v5.py --port 5005 --hardware esp32-omni
```

For original AMD_YES board mode:

```bash
python3 jetson_udp_receiver_v5.py --hardware amd-yes-v3 --amd-yes-root ~/Desktop/AMD_YES
```

### 3) Keyboard omni control with ultrasonic safety

```bash
python3 omni.py --motor-port /dev/ttyUSB0 --ultra-port /dev/ttyACM0
```

Controls: `W/A/S/D` strafe/drive, `Q/E` rotate, `Space` stop, `X` quit.

### 4) IMU serial publisher

```bash
python3 imu_Serial.py --backend auto --shared-file /tmp/imu_mag_latest.json
```

### 5) Cube face detector

```bash
python3 test016.py --source 0 --show_all
```

## Notes

- Serial port defaults in scripts are hardware-specific and may need overrides.
- Many runtime options are available via `--help` on each script.
- `test016.py` is compute-heavy and expects a working camera + ML runtime.

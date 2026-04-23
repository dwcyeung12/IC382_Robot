#!/usr/bin/env python3
"""Read an Xbox controller and send full state over UDP."""

from __future__ import annotations

import argparse
import json
import socket
import sys

import pygame


DEFAULT_AXIS_MAP = {
    "leftx": 0,
    "lefty": 1,
    "rightx": 2,
    "righty": 3,
    "triggerleft": 4,
    "triggerright": 5,
}

DEFAULT_BUTTON_MAP = {
    "a": 0,
    "b": 1,
    "x": 2,
    "y": 3,
    "back": 4,
    "guide": 5,
    "start": 6,
    "leftstick": 7,
    "rightstick": 8,
    "leftshoulder": 9,
    "rightshoulder": 10,
    "dpad_up": 11,
    "dpad_down": 12,
    "dpad_left": 13,
    "dpad_right": 14,
}

CONTROLLER_PROFILES = {
    "Xbox Series X Controller": {
        "axis_map": {
            "leftx": 0,
            "lefty": 1,
            "rightx": 2,
            "righty": 3,
            "triggerleft": 4,
            "triggerright": 5,
        },
        "button_map": {
            "a": 0,
            "b": 1,
            "x": 2,
            "y": 3,
            "back": 4,
            "guide": 5,
            "start": 6,
            "leftstick": 7,
            "rightstick": 8,
            "leftshoulder": 9,
            "rightshoulder": 10,
            "dpad_up": 11,
            "dpad_down": 12,
            "dpad_left": 13,
            "dpad_right": 14,
        },
    },
    "Controller": {
        "axis_map": dict(DEFAULT_AXIS_MAP),
        "button_map": dict(DEFAULT_BUTTON_MAP),
    },
}

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

BACKGROUND = (244, 246, 241)
PANEL = (255, 255, 255)
PANEL_BORDER = (198, 205, 191)
TEXT = (34, 43, 31)
TEXT_MUTED = (95, 108, 88)
ACTIVE = (71, 142, 92)
ACTIVE_TEXT = (255, 255, 255)
INACTIVE = (229, 233, 224)
VIRTUAL = (233, 180, 78)
BAR_NEG = (211, 112, 92)
BAR_POS = (66, 150, 113)
BAR_TRACK = (223, 228, 218)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send full joystick state to a Jetson over UDP."
    )
    parser.add_argument("--jetson-ip", default="10.11.31.236", help="Jetson IP")
    parser.add_argument("--port", type=int, default=5005, help="UDP port")
    parser.add_argument("--interval", type=float, default=0.02, help="Send interval")
    parser.add_argument("--joystick-index", type=int, default=0, help="Device index")
    parser.add_argument("--deadzone", type=float, default=0.08, help="Stick deadzone")
    parser.add_argument("--scale", type=float, default=1.0, help="Drive scale")
    parser.add_argument("--leftx-axis", type=int, default=None, help="Override axis index for left stick X")
    parser.add_argument("--lefty-axis", type=int, default=None, help="Override axis index for left stick Y")
    parser.add_argument("--rightx-axis", type=int, default=None, help="Override axis index for right stick X")
    parser.add_argument("--righty-axis", type=int, default=None, help="Override axis index for right stick Y")
    parser.add_argument("--triggerleft-axis", type=int, default=None, help="Override axis index for left trigger")
    parser.add_argument("--triggerright-axis", type=int, default=None, help="Override axis index for right trigger")
    parser.add_argument("--verbose", action="store_true", help="Print full state")
    return parser.parse_args()


def apply_deadzone(value: float, deadzone: float) -> float:
    return 0.0 if abs(value) < deadzone else value


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def get_controller_profile(controller_name: str) -> dict[str, dict[str, int]]:
    return CONTROLLER_PROFILES.get(
        controller_name,
        {
            "axis_map": dict(DEFAULT_AXIS_MAP),
            "button_map": dict(DEFAULT_BUTTON_MAP),
        },
    )


def get_axis_map(args: argparse.Namespace, controller_name: str) -> dict[str, int]:
    axis_map = dict(get_controller_profile(controller_name)["axis_map"])
    overrides = {
        "leftx": args.leftx_axis,
        "lefty": args.lefty_axis,
        "rightx": args.rightx_axis,
        "righty": args.righty_axis,
        "triggerleft": args.triggerleft_axis,
        "triggerright": args.triggerright_axis,
    }
    for name, value in overrides.items():
        if value is not None:
            axis_map[name] = value
    return axis_map


def get_button_map(controller_name: str) -> dict[str, int]:
    return dict(get_controller_profile(controller_name)["button_map"])


def map_xbox_axes(axes: list[float], axis_map: dict[str, int]) -> dict[str, float]:
    return {
        name: axes[index]
        for name, index in axis_map.items()
        if index < len(axes)
    }


def map_xbox_buttons(buttons: list[int], button_map: dict[str, int]) -> dict[str, int]:
    mapped = {
        name: buttons[index]
        for name, index in button_map.items()
        if index < len(buttons)
    }
    mapped.setdefault("dpad_up", 0)
    mapped.setdefault("dpad_down", 0)
    mapped.setdefault("dpad_left", 0)
    mapped.setdefault("dpad_right", 0)
    return mapped


def apply_hat_mapping(controller_buttons: dict[str, int], hats: list[list[int]]) -> dict[str, int]:
    mapped = dict(controller_buttons)
    if hats:
        x, y = hats[0]
        mapped["dpad_up"] = 1 if y > 0 else 0
        mapped["dpad_down"] = 1 if y < 0 else 0
        mapped["dpad_left"] = 1 if x < 0 else 0
        mapped["dpad_right"] = 1 if x > 0 else 0
    return mapped


def build_payload(
    name: str,
    seq: int,
    axes: list[float],
    buttons: list[int],
    hats: list[list[int]],
    scale: float,
    axis_map: dict[str, int],
    button_map: dict[str, int],
) -> dict[str, object]:
    controller_axes = map_xbox_axes(axes, axis_map)
    controller_buttons = apply_hat_mapping(map_xbox_buttons(buttons, button_map), hats)
    forward = clamp(-apply_deadzone(controller_axes.get("lefty", 0.0), 0.0) * scale)
    turn = clamp(apply_deadzone(controller_axes.get("leftx", 0.0), 0.0) * scale)
    return {
        "type": "controller_state",
        "seq": seq,
        "name": name,
        "axes": axes,
        "buttons": buttons,
        "hats": hats,
        "controller_axes": controller_axes,
        "controller_buttons": controller_buttons,
        "drive": {"forward": round(forward, 3), "turn": round(turn, 3)},
    }


def draw_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = TEXT,
) -> None:
    bitmap = font.render(text, True, color)
    screen.blit(bitmap, position)


def draw_button_chip(
    screen: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    active: bool,
    virtual: bool,
) -> None:
    if active:
        fill = ACTIVE
        text_color = ACTIVE_TEXT
    elif virtual:
        fill = VIRTUAL
        text_color = TEXT
    else:
        fill = INACTIVE
        text_color = TEXT
    pygame.draw.rect(screen, fill, rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=1, border_radius=10)
    text_rect = font.render(label, True, text_color).get_rect(center=rect.center)
    screen.blit(font.render(label, True, text_color), text_rect)


def draw_axis_bar(
    screen: pygame.Surface,
    label_font: pygame.font.Font,
    value_font: pygame.font.Font,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
) -> None:
    draw_text(screen, label_font, label, (x, y))
    draw_text(screen, value_font, f"{value:+.3f}", (x + width - 70, y), TEXT_MUTED)

    bar_y = y + 24
    bar_rect = pygame.Rect(x, bar_y, width, 16)
    pygame.draw.rect(screen, BAR_TRACK, bar_rect, border_radius=8)

    center_x = x + width // 2
    pygame.draw.line(screen, PANEL_BORDER, (center_x, bar_y), (center_x, bar_y + 16), 2)

    clamped = max(-1.0, min(1.0, value))
    if clamped != 0.0:
        if clamped > 0:
            fill_width = int((width // 2) * clamped)
            fill_rect = pygame.Rect(center_x, bar_y, fill_width, 16)
            fill_color = BAR_POS
        else:
            fill_width = int((width // 2) * abs(clamped))
            fill_rect = pygame.Rect(center_x - fill_width, bar_y, fill_width, 16)
            fill_color = BAR_NEG
        pygame.draw.rect(screen, fill_color, fill_rect, border_radius=8)


def draw_ui(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    destination: tuple[str, int],
    controller_name: str,
    payload: dict[str, object] | None,
    active: bool,
    virtual_buttons: dict[str, int],
) -> dict[str, pygame.Rect]:
    screen.fill(BACKGROUND)

    header_rect = pygame.Rect(20, 20, 940, 92)
    pygame.draw.rect(screen, PANEL, header_rect, border_radius=16)
    pygame.draw.rect(screen, PANEL_BORDER, header_rect, width=1, border_radius=16)

    status_text = "CONNECTED" if active else "WAITING FOR CONTROLLER"
    status_color = ACTIVE if active else BAR_NEG
    draw_text(screen, title_font, "A05 Controller Teleop", (40, 38))
    draw_text(screen, body_font, f"Target: {destination[0]}:{destination[1]}", (40, 74), TEXT_MUTED)
    draw_text(screen, body_font, f"Controller: {controller_name}", (360, 38), TEXT_MUTED)
    draw_text(screen, body_font, status_text, (360, 74), status_color)

    axis_panel = pygame.Rect(20, 132, 450, 560)
    button_panel = pygame.Rect(490, 132, 470, 560)
    pygame.draw.rect(screen, PANEL, axis_panel, border_radius=16)
    pygame.draw.rect(screen, PANEL_BORDER, axis_panel, width=1, border_radius=16)
    pygame.draw.rect(screen, PANEL, button_panel, border_radius=16)
    pygame.draw.rect(screen, PANEL_BORDER, button_panel, width=1, border_radius=16)

    draw_text(screen, title_font, "Axes", (40, 152))
    draw_text(screen, title_font, "Buttons", (510, 152))

    if payload is None:
        draw_text(screen, body_font, "Move the controller or click a button tile to start.", (40, 200), TEXT_MUTED)
        draw_text(screen, small_font, "Left click toggles a software button. Right click clears all software buttons.", (510, 640), TEXT_MUTED)
        pygame.display.flip()
        return get_button_rects()

    controller_axes = payload["controller_axes"]
    controller_buttons = payload["controller_buttons"]
    drive = payload["drive"]

    draw_text(
        screen,
        body_font,
        f"forward={drive['forward']:+.3f}  turn={drive['turn']:+.3f}",
        (40, 196),
    )

    axis_labels = [
        ("leftx", controller_axes.get("leftx", 0.0)),
        ("lefty", controller_axes.get("lefty", 0.0)),
        ("rightx", controller_axes.get("rightx", 0.0)),
        ("righty", controller_axes.get("righty", 0.0)),
        ("triggerleft", controller_axes.get("triggerleft", 0.0)),
        ("triggerright", controller_axes.get("triggerright", 0.0)),
    ]
    for index, (label, value) in enumerate(axis_labels):
        draw_axis_bar(
            screen,
            body_font,
            small_font,
            40,
            240 + index * 68,
            410,
            label,
            float(value),
        )

    rows = 5
    cols = 3
    chip_width = 132
    chip_height = 62
    gap_x = 14
    gap_y = 14
    start_x = 510
    start_y = 210
    button_rects = get_button_rects()
    for name in BUTTON_ORDER:
        rect = button_rects[name]
        draw_button_chip(
            screen,
            body_font,
            rect,
            name,
            bool(controller_buttons.get(name, 0)),
            bool(virtual_buttons.get(name, 0)),
        )

    pygame.display.flip()
    return button_rects


def get_button_rects() -> dict[str, pygame.Rect]:
    rows = 5
    cols = 3
    chip_width = 132
    chip_height = 62
    gap_x = 14
    gap_y = 14
    start_x = 510
    start_y = 210
    rects: dict[str, pygame.Rect] = {}
    for index, name in enumerate(BUTTON_ORDER):
        row = index // cols
        col = index % cols
        rects[name] = pygame.Rect(
            start_x + col * (chip_width + gap_x),
            start_y + row * (chip_height + gap_y),
            chip_width,
            chip_height,
        )
    return rects


def merge_virtual_buttons(
    controller_buttons: dict[str, int],
    virtual_buttons: dict[str, int],
) -> dict[str, int]:
    merged = dict(controller_buttons)
    for name in BUTTON_ORDER:
        merged[name] = 1 if merged.get(name, 0) or virtual_buttons.get(name, 0) else 0
    return merged


def main() -> int:
    args = parse_args()

    pygame.init()
    pygame.joystick.init()
    screen = pygame.display.set_mode((980, 720))
    pygame.display.set_caption("IC382 Network Teleop Sender")
    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 38)
    body_font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 24)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = (args.jetson_ip, args.port)

    joysticks: dict[int, pygame.joystick.Joystick] = {}
    active_id: int | None = None
    seq = 0
    next_send_ms = 0
    last_name = "unknown"
    last_payload: dict[str, object] | None = None
    axis_map = get_axis_map(args, last_name)
    button_map = get_button_map(last_name)
    virtual_buttons = {name: 0 for name in BUTTON_ORDER}
    button_rects = get_button_rects()

    print(f"Sending UDP teleop to {destination[0]}:{destination[1]}")
    print("Keep the pygame window focused. Press Ctrl+C to stop.")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        for name, rect in button_rects.items():
                            if rect.collidepoint(event.pos):
                                virtual_buttons[name] = 1
                                print(f"\nSoftware button one-shot: {name}")
                                break
                    elif event.button == 3:
                        for name in BUTTON_ORDER:
                            virtual_buttons[name] = 0
                        print("\nSoftware buttons cleared.")

                if event.type == pygame.JOYDEVICEADDED:
                    joy = pygame.joystick.Joystick(event.device_index)
                    jid = joy.get_instance_id()
                    joysticks[jid] = joy
                    if len(joysticks) == 1 or event.device_index == args.joystick_index:
                        active_id = jid
                        last_name = joy.get_name()
                        axis_map = get_axis_map(args, last_name)
                        button_map = get_button_map(last_name)
                        print(
                            f"\nUsing controller: {last_name} "
                            f"axes={joy.get_numaxes()} buttons={joy.get_numbuttons()} hats={joy.get_numhats()}"
                        )
                        print(f"Axis map: {axis_map}")
                        print(f"Button map: {button_map}")

                if event.type == pygame.JOYDEVICEREMOVED:
                    joystick = joysticks.pop(event.instance_id, None)
                    if joystick is not None:
                        joystick.quit()
                    if active_id == event.instance_id:
                        active_id = next(iter(joysticks), None)
                        print("\nActive controller disconnected.")

                if event.type in (
                    pygame.JOYAXISMOTION,
                    pygame.JOYBUTTONDOWN,
                    pygame.JOYBUTTONUP,
                    pygame.JOYHATMOTION,
                ):
                    if event.instance_id in joysticks and active_id != event.instance_id:
                        active_id = event.instance_id
                        last_name = joysticks[active_id].get_name()
                        axis_map = get_axis_map(args, last_name)
                        button_map = get_button_map(last_name)
                        joystick = joysticks[active_id]
                        print(
                            f"\nSwitched active controller to: {last_name} "
                            f"instance_id={active_id} axes={joystick.get_numaxes()} "
                            f"buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}"
                        )
                        print(f"Axis map: {axis_map}")
                        print(f"Button map: {button_map}")

            button_rects = draw_ui(
                screen,
                title_font,
                body_font,
                small_font,
                destination,
                last_name,
                last_payload,
                active_id is not None and active_id in joysticks,
                virtual_buttons,
            )

            now_ms = pygame.time.get_ticks()
            if now_ms < next_send_ms:
                clock.tick(60)
                continue

            if active_id is not None and active_id in joysticks:
                joystick = joysticks[active_id]
                axes = [
                    round(apply_deadzone(joystick.get_axis(i), args.deadzone), 3)
                    for i in range(joystick.get_numaxes())
                ]
                buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
                hats = [list(joystick.get_hat(i)) for i in range(joystick.get_numhats())]
            else:
                axes = [0.0] * 6
                buttons = [0] * 15
                hats = []

            payload = build_payload(last_name, seq, axes, buttons, hats, args.scale, axis_map, button_map)
            payload["controller_buttons"] = merge_virtual_buttons(payload["controller_buttons"], virtual_buttons)
            last_payload = payload
            sock.sendto(json.dumps(payload).encode("utf-8"), destination)
            for name in BUTTON_ORDER:
                virtual_buttons[name] = 0

            if args.verbose:
                line = (
                    f"\rseq={seq} axes={axes} buttons={buttons} hats={hats} "
                    f"controller_axes={payload['controller_axes']} controller_buttons={payload['controller_buttons']} "
                    f"drive={payload['drive']}"
                )
            else:
                pressed = [name for name, value in payload["controller_buttons"].items() if value]
                line = (
                    f"\rforward={payload['drive']['forward']:+.3f} "
                    f"turn={payload['drive']['turn']:+.3f} pressed={pressed}"
                )
            print(line, end="", flush=True)

            seq += 1
            next_send_ms = now_ms + max(1, int(args.interval * 1000))
            clock.tick(60)
    except KeyboardInterrupt:
        print("\nStopping teleop sender.")
        zero_axes = [0.0] * 6
        zero_buttons = [0] * 15
        zero_hats: list[list[int]] = []
        payload = build_payload(last_name, -1, zero_axes, zero_buttons, zero_hats, 1.0, axis_map, button_map)
        sock.sendto(json.dumps(payload).encode("utf-8"), destination)
        return 0
    finally:
        sock.close()
        for joystick in joysticks.values():
            joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())

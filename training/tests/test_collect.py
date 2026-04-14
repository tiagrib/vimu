"""Tests for collect_pose.py controller and utilities."""

import json
import threading
import time

import pytest

from collect_pose import (
    WebSocketController,
    generate_poses,
    load_calibration,
)


# ─── Calibration loader ──────────────────────────────────────────────────────


def test_load_calibration(tmp_path):
    """load_calibration parses a TOML file and returns joint ranges."""
    toml = tmp_path / "cal.toml"
    toml.write_text(
        'robot_name = "test"\n'
        'created_at = "unix:0"\n'
        'notes = ""\n'
        "\n"
        "[[joints]]\n"
        'name = "J1"\n'
        "center_pwm = 1500\n"
        "pwm_per_rad_pos = 500.0\n"
        "pwm_per_rad_neg = 500.0\n"
        "min_rad = -1.0\n"
        "max_rad = 1.0\n"
        "inverted = false\n"
        "\n"
        "[[joints]]\n"
        'name = "J2"\n'
        "center_pwm = 1500\n"
        "pwm_per_rad_pos = 500.0\n"
        "pwm_per_rad_neg = 500.0\n"
        "min_rad = -0.5\n"
        "max_rad = 0.8\n"
        "inverted = true\n"
    )
    joints = load_calibration(str(toml))
    assert len(joints) == 2
    assert joints[0]["name"] == "J1"
    assert joints[0]["min_rad"] == -1.0
    assert joints[0]["max_rad"] == 1.0
    assert joints[1]["name"] == "J2"
    assert joints[1]["min_rad"] == -0.5
    assert joints[1]["max_rad"] == 0.8


def test_load_calibration_empty_raises(tmp_path):
    """load_calibration raises on a file with no joints."""
    toml = tmp_path / "empty.toml"
    toml.write_text('robot_name = "test"\ncreated_at = "unix:0"\nnotes = ""\n')
    with pytest.raises(ValueError, match="No joints"):
        load_calibration(str(toml))


# ─── WebSocket Controller ────────────────────────────────────────────────────


def _start_mock_ws_server():
    """Start a mock WebSocket server, return (server, port, received_list)."""
    from websockets.sync.server import serve

    received = []

    def handler(websocket):
        for msg in websocket:
            received.append(json.loads(msg))
            websocket.send(json.dumps({
                "type": "state",
                "positions": [0, 0, 0],
            }))

    server = serve(handler, "localhost", 0)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port, received


def test_websocket_controller_connects_and_sends():
    """WebSocketController sends a proper JSON command via set_angles."""
    server, port, received = _start_mock_ws_server()
    try:
        ctrl = WebSocketController(f"ws://localhost:{port}", num_joints=3)
        ctrl.set_angles([0.1, 0.2, 0.3])
        time.sleep(0.2)

        assert len(received) >= 1
        msg = received[0]
        assert msg["type"] == "command"
        assert msg["positions"] == [0.1, 0.2, 0.3]

        ctrl.ws.close()
    finally:
        server.shutdown()


def test_websocket_controller_close_sends_neutral():
    """close() sends a command with all-zero positions."""
    server, port, received = _start_mock_ws_server()
    try:
        ctrl = WebSocketController(f"ws://localhost:{port}", num_joints=3)
        ctrl.close()
        time.sleep(0.2)

        neutrals = [m for m in received if m["positions"] == [0.0, 0.0, 0.0]]
        assert len(neutrals) >= 1
    finally:
        server.shutdown()


# ─── Pose generation ─────────────────────────────────────────────────────────


def test_generate_poses():
    """generate_poses returns correct count with values in range."""
    joint_ranges = [
        {"name": "J1", "min_rad": -1.0, "max_rad": 1.0},
        {"name": "J2", "min_rad": -0.5, "max_rad": 0.8},
        {"name": "J3", "min_rad": -1.2, "max_rad": 1.2},
    ]
    poses = generate_poses(joint_ranges, num_poses=10)

    assert len(poses) == 10
    for pose in poses:
        assert len(pose) == 3
        for i, val in enumerate(pose):
            lo = joint_ranges[i]["min_rad"]
            hi = joint_ranges[i]["max_rad"]
            assert lo <= val <= hi


def test_generate_poses_deterministic():
    """Same seed produces identical poses."""
    jr = [{"name": "J1", "min_rad": -1.0, "max_rad": 1.0}]
    a = generate_poses(jr, 5, seed=99)
    b = generate_poses(jr, 5, seed=99)
    assert a == b

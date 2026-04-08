"""Tests for collect.py controller abstractions and utilities."""

import json
import threading
import time
import types

import pytest

from collect import (
    DEFAULT_JOINT_RANGES,
    RobotController,
    SerialController,
    WebSocketController,
    create_controller,
    generate_poses,
)


# ─── 1. Abstract base class ────────────────────────────────────────────────


def test_robot_controller_is_abstract():
    """RobotController cannot be instantiated directly."""
    with pytest.raises(TypeError):
        RobotController()


# ─── 2 & 3. Subclass checks ────────────────────────────────────────────────


def test_serial_controller_implements_interface():
    """SerialController is a subclass of RobotController."""
    assert issubclass(SerialController, RobotController)


def test_websocket_controller_implements_interface():
    """WebSocketController is a subclass of RobotController."""
    assert issubclass(WebSocketController, RobotController)


# ─── 4. WebSocket connect & send ────────────────────────────────────────────


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


# ─── 5. close() sends neutral ──────────────────────────────────────────────


def test_websocket_controller_close_sends_neutral():
    """close() sends a command with all-zero positions."""
    server, port, received = _start_mock_ws_server()
    try:
        ctrl = WebSocketController(f"ws://localhost:{port}", num_joints=3)
        ctrl.close()
        time.sleep(0.2)

        # The last command should be the neutral pose
        neutrals = [m for m in received if m["positions"] == [0.0, 0.0, 0.0]]
        assert len(neutrals) >= 1
    finally:
        server.shutdown()


# ─── 6. create_controller with --ws ────────────────────────────────────────


def test_create_controller_with_ws():
    """create_controller returns WebSocketController when ws arg is set."""
    server, port, _ = _start_mock_ws_server()
    try:
        args = types.SimpleNamespace(ws=f"ws://localhost:{port}", num_joints=3)
        ctrl = create_controller(args)
        assert isinstance(ctrl, WebSocketController)
        ctrl.ws.close()
    finally:
        server.shutdown()


# ─── 7. create_controller serial without port ──────────────────────────────


def test_create_controller_with_serial_raises_without_port():
    """create_controller raises when the serial port does not exist."""
    args = types.SimpleNamespace(
        ws=None,
        serial="/dev/ttyNONEXISTENT999",
        baud=500000,
        num_joints=6,
    )
    with pytest.raises(Exception):
        create_controller(args)


# ─── 8. generate_poses ─────────────────────────────────────────────────────


def test_generate_poses():
    """generate_poses returns correct count with values in range."""
    num_joints = 3
    num_poses = 10
    poses = generate_poses(num_joints, num_poses)

    assert len(poses) == num_poses
    for pose in poses:
        assert len(pose) == num_joints
        for i, val in enumerate(pose):
            lo, hi = DEFAULT_JOINT_RANGES[i]
            assert lo <= val <= hi, f"Joint {i} value {val} out of range [{lo}, {hi}]"

"""
Tests for the VIMU Arduino binary serial protocol.

Validates frame encoding, checksum computation, and edge cases
without requiring an actual serial port or Arduino hardware.

Protocol:
  TX (Host -> Arduino): [0xAA] [cmd] [len] [payload...] [checksum]
  RX (Arduino -> Host):  [0xBB] [status] [len] [payload...] [checksum]
  Checksum = XOR of cmd/status, len, and all payload bytes
"""

import struct

# ── Protocol constants (mirrored from both Arduino and Python sides) ─────────

FRAME_START_TX = 0xAA  # Host -> Arduino
FRAME_START_RX = 0xBB  # Arduino -> Host

CMD_SET_POSITIONS = 0x01
CMD_QUERY_STATUS = 0x02
CMD_DETACH_ALL = 0x04

STATUS_OK = 0x00
STATUS_ERROR = 0x01

SERVO_MIN_US = 500
SERVO_MAX_US = 2500

NUM_SERVOS = 6


# ── Helper functions that replicate the protocol logic ───────────────────────


def compute_checksum(cmd_or_status: int, payload: bytes) -> int:
    """Compute checksum as XOR of cmd/status, length, and all payload bytes."""
    checksum = cmd_or_status ^ len(payload)
    for b in payload:
        checksum ^= b
    return checksum


def encode_tx_frame(cmd: int, payload: bytes = b"") -> bytes:
    """Encode a TX frame (host -> Arduino), matching ServoController._send()."""
    frame = bytes([FRAME_START_TX, cmd, len(payload)]) + payload
    checksum = 0
    for b in frame[1:]:
        checksum ^= b
    frame += bytes([checksum])
    return frame


def encode_rx_frame(status: int, payload: bytes = b"") -> bytes:
    """Encode an RX frame (Arduino -> host), matching send_response()."""
    frame = bytes([FRAME_START_RX, status, len(payload)]) + payload
    checksum = status ^ len(payload)
    for b in payload:
        checksum ^= b
    frame += bytes([checksum])
    return frame


def parse_tx_frame(data: bytes):
    """Parse a TX frame and return (cmd, payload) or None on failure."""
    if len(data) < 4:
        return None
    if data[0] != FRAME_START_TX:
        return None
    cmd = data[1]
    length = data[2]
    if len(data) != 3 + length + 1:
        return None
    payload = data[3 : 3 + length]
    checksum = data[3 + length]
    expected = compute_checksum(cmd, payload)
    if checksum != expected:
        return None
    return cmd, payload


def parse_rx_frame(data: bytes):
    """Parse an RX frame and return (status, payload) or None on failure."""
    if len(data) < 4:
        return None
    if data[0] != FRAME_START_RX:
        return None
    status = data[1]
    length = data[2]
    if len(data) != 3 + length + 1:
        return None
    payload = data[3 : 3 + length]
    checksum = data[3 + length]
    expected = compute_checksum(status, payload)
    if checksum != expected:
        return None
    return status, payload


def make_set_positions_payload(microseconds: list[int]) -> bytes:
    """Build SET_POSITIONS payload from a list of pulse widths in us."""
    payload = b""
    for us in microseconds:
        payload += struct.pack("<H", us)
    return payload


# ── Tests ────────────────────────────────────────────────────────────────────


class TestChecksumComputation:
    """Verify that checksum logic is consistent across encode/decode."""

    def test_checksum_zero_payload(self):
        """Checksum with empty payload is just cmd ^ 0."""
        assert compute_checksum(0x04, b"") == 0x04 ^ 0x00

    def test_checksum_single_byte_payload(self):
        cs = compute_checksum(0x01, b"\x05")
        assert cs == 0x01 ^ 0x01 ^ 0x05  # cmd ^ len ^ payload

    def test_checksum_multi_byte_payload(self):
        payload = b"\x10\x20\x30"
        cs = compute_checksum(0x02, payload)
        expected = 0x02 ^ 3 ^ 0x10 ^ 0x20 ^ 0x30
        assert cs == expected

    def test_checksum_matches_between_encode_and_parse_tx(self):
        """A frame encoded with encode_tx_frame should parse back correctly."""
        payload = struct.pack("<HH", 1500, 2000)
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)
        result = parse_tx_frame(frame)
        assert result is not None
        cmd, parsed_payload = result
        assert cmd == CMD_SET_POSITIONS
        assert parsed_payload == payload

    def test_checksum_matches_between_encode_and_parse_rx(self):
        """A frame encoded with encode_rx_frame should parse back correctly."""
        payload = struct.pack("<HH", 1500, 2000)
        frame = encode_rx_frame(STATUS_OK, payload)
        result = parse_rx_frame(frame)
        assert result is not None
        status, parsed_payload = result
        assert status == STATUS_OK
        assert parsed_payload == payload

    def test_corrupted_checksum_fails_parse(self):
        frame = encode_tx_frame(CMD_DETACH_ALL, b"")
        corrupted = frame[:-1] + bytes([frame[-1] ^ 0xFF])
        assert parse_tx_frame(corrupted) is None


class TestSetPositionsFrame:
    """Test encoding of SET_POSITIONS command frames."""

    def test_single_servo(self):
        payload = struct.pack("<H", 1500)
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)

        assert frame[0] == FRAME_START_TX
        assert frame[1] == CMD_SET_POSITIONS
        assert frame[2] == 2  # len = 2 bytes for one uint16
        assert frame[3:5] == payload
        assert parse_tx_frame(frame) == (CMD_SET_POSITIONS, payload)

    def test_six_servos(self):
        positions = [500, 1000, 1500, 2000, 2500, 1234]
        payload = make_set_positions_payload(positions)
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)

        assert frame[2] == 12  # 6 servos * 2 bytes each
        result = parse_tx_frame(frame)
        assert result is not None
        _, parsed_payload = result
        for i, expected_us in enumerate(positions):
            actual_us = struct.unpack_from("<H", parsed_payload, i * 2)[0]
            assert actual_us == expected_us

    def test_max_servos_payload_length(self):
        """NUM_SERVOS servos should produce a 12-byte payload."""
        positions = [1500] * NUM_SERVOS
        payload = make_set_positions_payload(positions)
        assert len(payload) == NUM_SERVOS * 2

    def test_min_pulse_width(self):
        payload = struct.pack("<H", SERVO_MIN_US)
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)
        _, parsed = parse_tx_frame(frame)
        assert struct.unpack("<H", parsed)[0] == SERVO_MIN_US

    def test_max_pulse_width(self):
        payload = struct.pack("<H", SERVO_MAX_US)
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)
        _, parsed = parse_tx_frame(frame)
        assert struct.unpack("<H", parsed)[0] == SERVO_MAX_US


class TestQueryStatusFrame:
    """Test encoding of QUERY_STATUS command and response."""

    def test_query_request_has_no_payload(self):
        frame = encode_tx_frame(CMD_QUERY_STATUS, b"")
        assert frame[2] == 0  # len = 0
        assert len(frame) == 4  # start + cmd + len + checksum

    def test_query_response_contains_positions(self):
        """Arduino responds with 6 uint16_le position values."""
        positions = [1500, 1500, 1500, 1500, 1500, 1500]
        payload = b""
        for us in positions:
            payload += struct.pack("<H", us)
        frame = encode_rx_frame(STATUS_OK, payload)

        result = parse_rx_frame(frame)
        assert result is not None
        status, parsed = result
        assert status == STATUS_OK
        assert len(parsed) == NUM_SERVOS * 2
        for i in range(NUM_SERVOS):
            assert struct.unpack_from("<H", parsed, i * 2)[0] == 1500


class TestDetachAllFrame:
    """Test DETACH_ALL command frame."""

    def test_detach_frame_structure(self):
        frame = encode_tx_frame(CMD_DETACH_ALL, b"")
        assert frame[0] == FRAME_START_TX
        assert frame[1] == CMD_DETACH_ALL
        assert frame[2] == 0  # no payload
        assert len(frame) == 4

    def test_detach_checksum(self):
        frame = encode_tx_frame(CMD_DETACH_ALL, b"")
        # checksum = CMD_DETACH_ALL ^ 0 = 0x04
        assert frame[3] == CMD_DETACH_ALL

    def test_detach_response_ok(self):
        frame = encode_rx_frame(STATUS_OK, b"")
        result = parse_rx_frame(frame)
        assert result is not None
        status, payload = result
        assert status == STATUS_OK
        assert payload == b""


class TestEdgeCases:
    """Protocol edge cases."""

    def test_zero_length_payload(self):
        """Zero-length payload should still produce valid frame."""
        frame = encode_tx_frame(0x02, b"")
        assert len(frame) == 4
        assert parse_tx_frame(frame) is not None

    def test_frame_too_short_to_parse(self):
        assert parse_tx_frame(b"\xAA\x01") is None
        assert parse_rx_frame(b"\xBB") is None

    def test_wrong_start_byte(self):
        frame = encode_tx_frame(CMD_SET_POSITIONS, b"\x00\x01")
        # Replace start byte
        bad = bytes([0xCC]) + frame[1:]
        assert parse_tx_frame(bad) is None

    def test_length_mismatch_fails(self):
        """If the frame is truncated, parsing should fail."""
        payload = struct.pack("<HHH", 1000, 1500, 2000)
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)
        truncated = frame[:-2]  # Remove checksum and last payload byte
        assert parse_tx_frame(truncated) is None

    def test_max_payload_64_bytes(self):
        """Arduino RX buffer is 64 bytes; test that 64-byte payload encodes."""
        payload = bytes(range(64))
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)
        result = parse_tx_frame(frame)
        assert result is not None
        assert result[1] == payload

    def test_payload_exceeding_buffer_is_valid_at_protocol_level(self):
        """A 65-byte payload is valid protocol but would be rejected by Arduino
        state machine (len > RX_BUF_SIZE -> drops to WAIT_START).
        At the pure protocol encoding level, it should still encode/decode."""
        payload = bytes(range(65))
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)
        result = parse_tx_frame(frame)
        assert result is not None

    def test_unknown_command_still_produces_valid_frame(self):
        frame = encode_tx_frame(0xFF, b"")
        result = parse_tx_frame(frame)
        assert result is not None
        assert result[0] == 0xFF

    def test_all_zeros_payload(self):
        payload = b"\x00" * 12
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)
        result = parse_tx_frame(frame)
        assert result is not None
        assert result[1] == payload

    def test_all_0xff_payload(self):
        payload = b"\xFF" * 12
        frame = encode_tx_frame(CMD_SET_POSITIONS, payload)
        result = parse_tx_frame(frame)
        assert result is not None
        assert result[1] == payload


class TestPythonSendMethod:
    """Verify that our encode_tx_frame matches what ServoController._send() produces.

    We replicate the exact logic from collect.py's _send method and compare."""

    @staticmethod
    def _send_replica(cmd: int, payload: bytes = b"") -> bytes:
        """Exact replica of ServoController._send() logic, returning bytes."""
        frame = bytes([FRAME_START_TX, cmd, len(payload)]) + payload
        checksum = 0
        for b in frame[1:]:
            checksum ^= b
        frame += bytes([checksum])
        return frame

    def test_send_matches_encode_for_set_positions(self):
        payload = make_set_positions_payload([1500, 2000, 500])
        assert self._send_replica(CMD_SET_POSITIONS, payload) == encode_tx_frame(
            CMD_SET_POSITIONS, payload
        )

    def test_send_matches_encode_for_detach(self):
        assert self._send_replica(CMD_DETACH_ALL) == encode_tx_frame(
            CMD_DETACH_ALL, b""
        )

    def test_send_matches_encode_for_query(self):
        assert self._send_replica(CMD_QUERY_STATUS) == encode_tx_frame(
            CMD_QUERY_STATUS, b""
        )

    def test_send_set_positions_exact_bytes(self):
        """Verify the exact byte sequence for a known input."""
        payload = struct.pack("<H", 1500)  # 0xDC05 little-endian
        frame = self._send_replica(CMD_SET_POSITIONS, payload)

        assert frame[0] == 0xAA  # start
        assert frame[1] == 0x01  # CMD_SET_POSITIONS
        assert frame[2] == 0x02  # length
        assert frame[3] == 0xDC  # 1500 & 0xFF
        assert frame[4] == 0x05  # 1500 >> 8
        expected_checksum = 0x01 ^ 0x02 ^ 0xDC ^ 0x05
        assert frame[5] == expected_checksum

    def test_send_detach_exact_bytes(self):
        frame = self._send_replica(CMD_DETACH_ALL)
        assert frame == bytes([0xAA, 0x04, 0x00, 0x04])


class TestResponseFrames:
    """Test Arduino -> Host response frames."""

    def test_ok_response_no_payload(self):
        frame = encode_rx_frame(STATUS_OK, b"")
        assert frame == bytes([0xBB, 0x00, 0x00, 0x00])

    def test_error_response_no_payload(self):
        frame = encode_rx_frame(STATUS_ERROR, b"")
        # checksum = 0x01 ^ 0x00 = 0x01
        assert frame == bytes([0xBB, 0x01, 0x00, 0x01])

    def test_ok_response_with_position_payload(self):
        positions = [500, 1000, 1500, 2000, 2500, 1234]
        payload = b""
        for us in positions:
            payload += struct.pack("<H", us)
        frame = encode_rx_frame(STATUS_OK, payload)

        result = parse_rx_frame(frame)
        assert result is not None
        status, data = result
        assert status == STATUS_OK
        decoded = [struct.unpack_from("<H", data, i * 2)[0] for i in range(6)]
        assert decoded == positions

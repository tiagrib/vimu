# Serial Protocol Contract (Arduino ↔ Python)

## Transport
- Baud: 500000
- Format: Binary frames

## Frame Format
```
[START_BYTE: 0xAA] [CMD: u8] [LEN: u8] [PAYLOAD: LEN bytes] [CHECKSUM: u8]
```

Checksum: XOR of CMD + LEN + all PAYLOAD bytes.

## Commands

### SET_POSITIONS (0x01)
- **Direction**: Python → Arduino
- **Payload**: `num_servos` × `u16` (little-endian) microsecond pulse widths
- **Response**: None (fire-and-forget)

### QUERY_STATUS (0x02)
- **Direction**: Python → Arduino
- **Payload**: Empty
- **Response**: `[0xAA] [0x02] [LEN] [num_servos × u16 current_us] [checksum]`

### DETACH_ALL (0x04)
- **Direction**: Python → Arduino
- **Payload**: Empty
- **Response**: None

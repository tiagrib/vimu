# WebSocket Message Contract (Inference → Clients)

## Transport
- Protocol: WebSocket (text frames)
- Default endpoint: `ws://0.0.0.0:9001`

## Message Schema

```json
{
  "timestamp": <f64>,
  "fps": <f64>,
  "dims": [
    {
      "name": <string>,
      "raw": <f64>,
      "position": <f64>,
      "velocity": <f64>,
      "acceleration": <f64>
    }
  ]
}
```

### Fields

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `timestamp` | f64 | seconds | Time since inference start |
| `fps` | f64 | Hz | Measured frames per second |
| `dims[].name` | string | — | Dimension name: `joint_1`..`joint_N`, `base_roll`, `base_pitch` |
| `dims[].raw` | f64 | radians | Direct model output (noisy) |
| `dims[].position` | f64 | radians | EKF-filtered position |
| `dims[].velocity` | f64 | rad/s | EKF velocity estimate |
| `dims[].acceleration` | f64 | rad/s² | EKF acceleration estimate |

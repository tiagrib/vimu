/**
 * VIMU Data Collection Servo Controller
 * 
 * Used during training data collection to command servos
 * and confirm positions. Not used during inference.
 * 
 * Binary protocol at 500kbaud:
 *   Host → Arduino: [0xAA] [cmd] [len] [payload...] [checksum]
 *   Arduino → Host:  [0xBB] [status] [len] [payload...] [checksum]
 *   Checksum = XOR of all bytes after the start byte
 * 
 * Commands:
 *   0x01 SET_POSITIONS  - payload: N × uint16_le microseconds
 *   0x02 QUERY_STATUS   - responds with current servo positions
 *   0x04 DETACH_ALL     - release all servos
 */

// ─── Configuration (customize for your robot) ───────────────────────────────

#define NUM_SERVOS    6
#define SERIAL_BAUD   500000

const uint8_t SERVO_PINS[NUM_SERVOS] = {3, 5, 6, 9, 10, 11};

#define SERVO_MIN_US  500
#define SERVO_MAX_US  2500

// ─── Protocol constants ─────────────────────────────────────────────────────

#define FRAME_START_RX 0xAA
#define FRAME_START_TX 0xBB
#define CMD_SET_POSITIONS 0x01
#define CMD_QUERY_STATUS  0x02
#define CMD_DETACH_ALL    0x04
#define STATUS_OK    0x00
#define STATUS_ERROR 0x01

// ─── Servo state ────────────────────────────────────────────────────────────

volatile uint16_t current_us[NUM_SERVOS];
bool servos_attached = true;

void servos_init() {
    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
        pinMode(SERVO_PINS[i], OUTPUT);
        digitalWrite(SERVO_PINS[i], LOW);
        current_us[i] = 1500;
    }
}

void servos_update() {
    if (!servos_attached) return;
    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
        uint16_t pw = constrain(current_us[i], SERVO_MIN_US, SERVO_MAX_US);
        digitalWrite(SERVO_PINS[i], HIGH);
        delayMicroseconds(pw);
        digitalWrite(SERVO_PINS[i], LOW);
    }
}

// ─── Serial handling ────────────────────────────────────────────────────────

#define RX_BUF_SIZE 64
uint8_t rx_buf[RX_BUF_SIZE];
uint8_t rx_idx = 0;

enum RxState { WAIT_START, READ_CMD, READ_LEN, READ_PAYLOAD, READ_CHECKSUM };
RxState rx_state = WAIT_START;
uint8_t rx_cmd = 0, rx_len = 0, rx_checksum = 0;

void send_response(uint8_t status, const uint8_t* payload, uint8_t len) {
    Serial.write(FRAME_START_TX);
    Serial.write(status);
    Serial.write(len);
    uint8_t checksum = status ^ len;
    for (uint8_t i = 0; i < len; i++) {
        Serial.write(payload[i]);
        checksum ^= payload[i];
    }
    Serial.write(checksum);
}

void handle_command(uint8_t cmd, const uint8_t* payload, uint8_t len) {
    switch (cmd) {
        case CMD_SET_POSITIONS: {
            uint8_t num = len / 2;
            if (num > NUM_SERVOS) num = NUM_SERVOS;
            for (uint8_t i = 0; i < num; i++) {
                uint16_t us = payload[i*2] | (payload[i*2+1] << 8);
                current_us[i] = constrain(us, SERVO_MIN_US, SERVO_MAX_US);
            }
            servos_attached = true;
            send_response(STATUS_OK, NULL, 0);
            break;
        }
        case CMD_QUERY_STATUS: {
            uint8_t resp[NUM_SERVOS * 2];
            for (uint8_t i = 0; i < NUM_SERVOS; i++) {
                resp[i*2] = current_us[i] & 0xFF;
                resp[i*2+1] = (current_us[i] >> 8) & 0xFF;
            }
            send_response(STATUS_OK, resp, NUM_SERVOS * 2);
            break;
        }
        case CMD_DETACH_ALL: {
            servos_attached = false;
            for (uint8_t i = 0; i < NUM_SERVOS; i++)
                digitalWrite(SERVO_PINS[i], LOW);
            send_response(STATUS_OK, NULL, 0);
            break;
        }
        default:
            send_response(STATUS_ERROR, NULL, 0);
            break;
    }
}

void process_serial() {
    while (Serial.available()) {
        uint8_t b = Serial.read();
        switch (rx_state) {
            case WAIT_START:
                if (b == FRAME_START_RX) { rx_state = READ_CMD; rx_checksum = 0; }
                break;
            case READ_CMD:
                rx_cmd = b; rx_checksum ^= b; rx_state = READ_LEN;
                break;
            case READ_LEN:
                rx_len = b; rx_checksum ^= b; rx_idx = 0;
                rx_state = (rx_len == 0) ? READ_CHECKSUM :
                           (rx_len > RX_BUF_SIZE) ? WAIT_START : READ_PAYLOAD;
                break;
            case READ_PAYLOAD:
                rx_buf[rx_idx++] = b; rx_checksum ^= b;
                if (rx_idx >= rx_len) rx_state = READ_CHECKSUM;
                break;
            case READ_CHECKSUM:
                if (b == rx_checksum) handle_command(rx_cmd, rx_buf, rx_len);
                rx_state = WAIT_START;
                break;
        }
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

unsigned long last_update = 0;

void setup() {
    Serial.begin(SERIAL_BAUD);
    servos_init();
    pinMode(LED_BUILTIN, OUTPUT);
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_BUILTIN, HIGH); delay(100);
        digitalWrite(LED_BUILTIN, LOW);  delay(100);
    }
}

void loop() {
    process_serial();
    unsigned long now = millis();
    if (now - last_update >= 20) {
        last_update = now;
        servos_update();
    }
}

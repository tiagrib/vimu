# Glossary

| Term | Definition |
|------|-----------|
| **EKF** | Extended Kalman Filter — recursive state estimator that smooths noisy model output into position, velocity, and acceleration. |
| **ONNX** | Open Neural Network Exchange — portable model format used to transfer the trained PyTorch model to the Rust inference engine. |
| **proprioception** | Sense of body position; here, estimating robot joint angles and base orientation from vision alone. |
| **base orientation** | Roll and pitch of the robot's base/torso, estimated alongside joint angles. |
| **dim / dimension** | One output channel of the model — either a joint angle or a base orientation axis. |
| **process noise** | EKF tuning parameter — higher values make the filter more responsive to sudden changes but noisier. |
| **measurement noise** | EKF tuning parameter — lower values trust the model more; higher values smooth over model errors. |
| **settle time** | Delay after commanding a servo pose before capturing a frame, allowing mechanical vibrations to stop. |
| **sweep mode** | Automated data collection: Python commands random poses, waits for settle, captures frame + labels. |
| **tilted mode** | Manual data collection for base orientation: user physically tilts robot and enters angles. |

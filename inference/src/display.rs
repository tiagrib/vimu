#[cfg(feature = "camera")]
use opencv::core::{Mat, Point, Scalar};
#[cfg(feature = "camera")]
use opencv::imgproc;

#[cfg(feature = "camera")]
use crate::ekf::EkfState;

#[cfg(feature = "camera")]
pub struct DisplayRenderer {
    joint_ranges: Vec<(f64, f64)>,
    num_joints: usize,
}

#[cfg(feature = "camera")]
impl DisplayRenderer {
    pub fn new(dim_names: &[String], joint_ranges: Option<Vec<(f64, f64)>>) -> Self {
        let num_joints = dim_names.iter().filter(|n| n.starts_with("joint_")).count();
        let joint_ranges = joint_ranges.unwrap_or_else(|| {
            vec![(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2); num_joints]
        });
        Self {
            joint_ranges,
            num_joints,
        }
    }

    pub fn render(
        &self,
        frame: &Mat,
        state: &EkfState,
        names: &[String],
        fps: f64,
        latency_ms: f64,
        client_count: usize,
    ) -> opencv::Result<bool> {
        let mut display = frame.clone();
        self.draw_stats(&mut display, fps, latency_ms, client_count);
        self.draw_joint_bars(&mut display, state, names);
        self.draw_base_orientation(&mut display, state, names);

        opencv::highgui::imshow("VIMU", &display)?;
        Ok(opencv::highgui::wait_key(1)? == 'q' as i32)
    }

    fn draw_stats(&self, frame: &mut Mat, fps: f64, latency_ms: f64, clients: usize) {
        let color = if fps >= 50.0 {
            Scalar::new(0.0, 255.0, 0.0, 0.0)
        } else if fps >= 30.0 {
            Scalar::new(0.0, 255.0, 255.0, 0.0)
        } else {
            Scalar::new(0.0, 0.0, 255.0, 0.0)
        };
        let text = format!("{fps:.0} FPS | {latency_ms:.1}ms | {clients} clients");
        let _ = imgproc::put_text(
            frame,
            &text,
            Point::new(10, 20),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            imgproc::LINE_8,
            false,
        );
    }

    fn draw_joint_bars(&self, frame: &mut Mat, state: &EkfState, names: &[String]) {
        let mut joint_idx = 0usize;
        for (i, name) in names.iter().enumerate() {
            if !name.starts_with("joint_") {
                continue;
            }
            if joint_idx >= self.num_joints {
                break;
            }

            let (min_r, max_r) = self.joint_ranges[joint_idx];
            let y = 30 + joint_idx as i32 * 32;

            // White outline for full range
            let _ = imgproc::rectangle(
                frame,
                opencv::core::Rect::new(430, y, 200, 20),
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_8,
                0,
            );

            // Fill width based on position
            let pos = state.positions[i];
            let clamped = pos.clamp(min_r, max_r);
            let fill = ((clamped - min_r) / (max_r - min_r) * 200.0) as i32;
            let fill = fill.clamp(0, 200);

            // Color by velocity magnitude
            let vel_mag = state.velocities[i].abs();
            let bar_color = velocity_to_color(vel_mag);

            // Filled bar
            if fill > 0 {
                let _ = imgproc::rectangle(
                    frame,
                    opencv::core::Rect::new(430, y, fill, 20),
                    bar_color,
                    -1,
                    imgproc::LINE_8,
                    0,
                );
            }

            // Center tick mark
            let mid_x = 430 + 100;
            let _ = imgproc::line(
                frame,
                Point::new(mid_x, y),
                Point::new(mid_x, y + 20),
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_8,
                0,
            );

            // Label
            let label = format!("{name}: {pos:+.2}");
            let _ = imgproc::put_text(
                frame,
                &label,
                Point::new(300, y + 15),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.35,
                Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_8,
                false,
            );

            joint_idx += 1;
        }
    }

    fn draw_base_orientation(&self, frame: &mut Mat, state: &EkfState, names: &[String]) {
        let rows = frame.rows();
        let y = rows - 40;

        let roll_idx = names.iter().position(|n| n == "base_roll");
        let pitch_idx = names.iter().position(|n| n == "base_pitch");

        let roll = roll_idx.map(|i| state.positions[i]).unwrap_or(0.0);
        let pitch = pitch_idx.map(|i| state.positions[i]).unwrap_or(0.0);

        // Only draw if at least one base dim exists
        if roll_idx.is_none() && pitch_idx.is_none() {
            return;
        }

        let magnitude = roll.abs() + pitch.abs();
        let color = if magnitude < 0.1 {
            Scalar::new(0.0, 255.0, 0.0, 0.0)
        } else if magnitude < 0.3 {
            Scalar::new(0.0, 255.0, 255.0, 0.0)
        } else {
            Scalar::new(0.0, 0.0, 255.0, 0.0)
        };

        let roll_deg = roll.to_degrees();
        let pitch_deg = pitch.to_degrees();
        let text = format!("Roll: {roll_deg:+.1}\u{00b0} Pitch: {pitch_deg:+.1}\u{00b0}");
        let _ = imgproc::put_text(
            frame,
            &text,
            Point::new(10, y),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            imgproc::LINE_8,
            false,
        );
    }
}

/// Map velocity magnitude to a BGR color tuple.
#[allow(dead_code)]
fn velocity_to_color_rgb(vel_mag: f64) -> (f64, f64, f64) {
    if vel_mag < 0.5 {
        (255.0, 100.0, 0.0) // blue
    } else if vel_mag < 3.0 {
        (0.0, 200.0, 0.0) // green
    } else {
        (0.0, 0.0, 255.0) // red
    }
}

#[cfg(feature = "camera")]
fn velocity_to_color(vel_mag: f64) -> Scalar {
    let (b, g, r) = velocity_to_color_rgb(vel_mag);
    Scalar::new(b, g, r, 0.0)
}

/// Map a position within [min, max] to a pixel width in [0, max_px].
#[allow(dead_code)]
fn position_to_bar_width(pos: f64, min: f64, max: f64, max_px: i32) -> i32 {
    let clamped = pos.clamp(min, max);
    let ratio = (clamped - min) / (max - min);
    (ratio * max_px as f64) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_to_color() {
        // Low velocity -> blue
        let c = velocity_to_color_rgb(0.1);
        assert_eq!(c, (255.0, 100.0, 0.0));

        // Medium velocity -> green
        let c = velocity_to_color_rgb(1.0);
        assert_eq!(c, (0.0, 200.0, 0.0));

        // High velocity -> red
        let c = velocity_to_color_rgb(5.0);
        assert_eq!(c, (0.0, 0.0, 255.0));

        // Boundary: exactly 0.5 -> green
        let c = velocity_to_color_rgb(0.5);
        assert_eq!(c, (0.0, 200.0, 0.0));

        // Boundary: exactly 3.0 -> red
        let c = velocity_to_color_rgb(3.0);
        assert_eq!(c, (0.0, 0.0, 255.0));
    }

    #[test]
    fn test_position_to_bar_width() {
        // Midpoint
        assert_eq!(position_to_bar_width(0.0, -1.0, 1.0, 200), 100);

        // At minimum
        assert_eq!(position_to_bar_width(-1.0, -1.0, 1.0, 200), 0);

        // At maximum
        assert_eq!(position_to_bar_width(1.0, -1.0, 1.0, 200), 200);

        // Below minimum (clamped)
        assert_eq!(position_to_bar_width(-2.0, -1.0, 1.0, 200), 0);

        // Above maximum (clamped)
        assert_eq!(position_to_bar_width(5.0, -1.0, 1.0, 200), 200);

        // Quarter point
        assert_eq!(position_to_bar_width(-0.5, -1.0, 1.0, 200), 50);
    }
}

use nalgebra::{DMatrix, DVector};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct EkfState {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub accelerations: Vec<f64>,
    pub timestamp: f64,
}

/// Per-dimension constant-acceleration Kalman filter.
/// State per dimension: [position, velocity, acceleration].
pub struct PoseEkf {
    num_dims: usize,
    state_dim: usize,
    x: DVector<f64>,
    p: DMatrix<f64>,
    r: DMatrix<f64>,
    h: DMatrix<f64>,
    process_noise: f64,
    dt: f64,
    initialized: bool,
    last_time: Option<f64>,
}

impl PoseEkf {
    pub fn new(num_dims: usize, dt: f64, process_noise: f64, measurement_noise: f64) -> Self {
        let state_dim = 3 * num_dims;
        let x = DVector::zeros(state_dim);
        let p = DMatrix::identity(state_dim, state_dim);
        let r = DMatrix::identity(num_dims, num_dims) * measurement_noise;

        let mut h = DMatrix::zeros(num_dims, state_dim);
        for i in 0..num_dims {
            h[(i, 3 * i)] = 1.0;
        }

        Self {
            num_dims,
            state_dim,
            x,
            p,
            r,
            h,
            process_noise,
            dt,
            initialized: false,
            last_time: None,
        }
    }

    fn transition(&self, dt: f64) -> DMatrix<f64> {
        let mut f = DMatrix::identity(self.state_dim, self.state_dim);
        for i in 0..self.num_dims {
            let b = 3 * i;
            f[(b, b + 1)] = dt;
            f[(b, b + 2)] = 0.5 * dt * dt;
            f[(b + 1, b + 2)] = dt;
        }
        f
    }

    fn process_noise_mat(&self, dt: f64) -> DMatrix<f64> {
        let mut q = DMatrix::zeros(self.state_dim, self.state_dim);
        let qn = self.process_noise;
        for i in 0..self.num_dims {
            let b = 3 * i;
            let (d2, d3, d4, d5) = (dt * dt, dt.powi(3), dt.powi(4), dt.powi(5));
            q[(b, b)] = d5 / 20.0;
            q[(b, b + 1)] = d4 / 8.0;
            q[(b, b + 2)] = d3 / 6.0;
            q[(b + 1, b)] = d4 / 8.0;
            q[(b + 1, b + 1)] = d3 / 3.0;
            q[(b + 1, b + 2)] = d2 / 2.0;
            q[(b + 2, b)] = d3 / 6.0;
            q[(b + 2, b + 1)] = d2 / 2.0;
            q[(b + 2, b + 2)] = dt;
        }
        q * qn
    }

    pub fn update(&mut self, measurement: &[f64], timestamp: f64) -> EkfState {
        assert_eq!(measurement.len(), self.num_dims);
        let z = DVector::from_column_slice(measurement);

        if !self.initialized {
            for i in 0..self.num_dims {
                self.x[3 * i] = measurement[i];
            }
            self.initialized = true;
            self.last_time = Some(timestamp);
            return self.extract(timestamp);
        }

        let dt = self
            .last_time
            .map(|t| (timestamp - t).max(1e-6))
            .unwrap_or(self.dt);
        self.last_time = Some(timestamp);

        // Predict
        let f = self.transition(dt);
        let q = self.process_noise_mat(dt);
        self.x = &f * &self.x;
        self.p = &f * &self.p * f.transpose() + q;

        // Update
        let y = &z - &self.h * &self.x;
        let s = &self.h * &self.p * self.h.transpose() + &self.r;
        let s_inv = s.try_inverse().expect("Singular innovation covariance");
        let k = &self.p * self.h.transpose() * s_inv;

        self.x = &self.x + &k * &y;

        let eye = DMatrix::identity(self.state_dim, self.state_dim);
        let i_kh = &eye - &k * &self.h;
        self.p = &i_kh * &self.p * i_kh.transpose() + &k * &self.r * k.transpose();

        self.extract(timestamp)
    }

    fn extract(&self, timestamp: f64) -> EkfState {
        let mut pos = Vec::with_capacity(self.num_dims);
        let mut vel = Vec::with_capacity(self.num_dims);
        let mut acc = Vec::with_capacity(self.num_dims);
        for i in 0..self.num_dims {
            pos.push(self.x[3 * i]);
            vel.push(self.x[3 * i + 1]);
            acc.push(self.x[3 * i + 2]);
        }
        EkfState {
            positions: pos,
            velocities: vel,
            accelerations: acc,
            timestamp,
        }
    }
}

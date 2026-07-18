//! `ml-rs` — GPU-accelerated MaceLenia (Multi-channel Lenia) cellular automaton simulation.
//!
//! All computation (FFT, complex multiply, growth function, channel aggregation,
//! diffusion step) happens as GPU compute shaders via `wgpu`. Channel data is only
//! read back to the CPU when explicitly requested (for display or export).
//!
//! See [`orchestrator::GpuMaceLenia`] for the main simulation entry point.

pub mod config;
pub mod orchestrator;
pub mod wfft;

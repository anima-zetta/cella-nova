//! `Lenia_ca` — GPU-accelerated Flow Lenia cellular automaton simulation.
//!
//! All computation (FFT, complex multiply, growth function, channel aggregation,
//! Sobel gradients, flow field, reintegration tracking)
//! happens as GPU compute shaders via `wgpu`. Channel data is only read back
//! to the CPU when explicitly requested (for display or export).
//!
//! See [`orchestrator::GpuFlowLenia`] for the main simulation entry point.

pub mod orchestrator;
pub mod wfft;

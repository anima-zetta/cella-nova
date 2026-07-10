//! Shared wgpu context for GPU compute operations.

// ---------------------------------------------------------------------------
// WgpuContext
// ---------------------------------------------------------------------------

/// Manages the wgpu device and queue for GPU compute operations.
///
/// Create one instance and share it via `Arc` across all GPU resources.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuContext {
    /// Initializes a wgpu device and queue.
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter found!");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WgpuContext Device"),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        ))
        .expect("Failed to request GPU device!");

        Self { device, queue }
    }

    /// Wraps an existing device + queue pair.
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self { device, queue }
    }
}

impl Default for WgpuContext {
    fn default() -> Self {
        Self::new()
    }
}

//! Audio synthesis for MaceLenia — sparse, tonal sonar pings that react
//! to meaningful pattern changes on the grid.
//!
//! Design:
//! - No continuous noise, no ambient hiss, no rumble
//! - Each channel produces clean tonal pings at a distinct frequency
//! - Pings fire when spatial variance changes (mass redistribution)
//! - Per-channel cooldowns keep the soundscape sparse but responsive
//! - Frequency sweeps (chirps) + low-pass filter + delay for alien/underwater feel
//! - First frame is skipped to avoid the initial loud spike
//!
//! Features are extracted from the already-downloaded RGB render buffer
//! (zero additional GPU cost).

use std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum simultaneous pings (polyphony)
const MAX_PINGS: usize = 10;

/// Ping duration range in samples at 44100 Hz
const PING_MIN_SAMPLES: usize = 250; // ~6 ms — short blip
const PING_MAX_SAMPLES: usize = 3000; // ~68 ms — sonar ping

/// Frames between pings per channel
const COOLDOWN_MIN: usize = 2;
const COOLDOWN_MAX: usize = 10;

/// Variance change threshold
const VAR_CHANGE_THRESHOLD: f32 = 0.000_001;

/// Delay line for underwater echo
const DELAY_BUFFER_SIZE: usize = 16384;
const DELAY_TIME_SAMPLES: usize = 6615; // 150 ms
const DELAY_FEEDBACK: f32 = 0.35;
const DELAY_WET: f32 = 0.30;

/// One-pole low-pass filter cutoff (Hz) — lower = more muffled/underwater
const LP_CUTOFF: f32 = 1400.0;

/// Tremolo rate (Hz) for alien pulsating effect
const TREMOLO_RATE: f32 = 6.0;

// ---------------------------------------------------------------------------
// AudioFeatures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub channel_means: [f32; 3],
    pub channel_variances: [f32; 3],
}

// ---------------------------------------------------------------------------
// Ping — a single tonal event with frequency sweep
// ---------------------------------------------------------------------------

struct Ping {
    /// Current oscillator phase
    phase: f32,
    /// Starting frequency (Hz)
    freq_start: f32,
    /// Ending frequency (Hz) — ping sweeps from start to end
    freq_end: f32,
    /// Samples remaining
    remaining: usize,
    /// Total duration for envelope
    total_duration: usize,
    /// Peak amplitude [0, 1]
    amplitude: f32,
    /// Stereo pan [0 = left, 1 = right]
    pan: f32,
}

impl Ping {
    fn is_alive(&self) -> bool {
        self.remaining > 0
    }

    /// Current frequency at this point in the ping's lifetime (linear sweep).
    fn current_freq(&self) -> f32 {
        let t = 1.0 - (self.remaining as f32 / self.total_duration as f32);
        self.freq_start + (self.freq_end - self.freq_start) * t
    }
}

// ---------------------------------------------------------------------------
// Per-channel state
// ---------------------------------------------------------------------------

struct ChannelState {
    cooldown: usize,
    prev_variance: f32,
    base_freq: f32,
}

// ---------------------------------------------------------------------------
// AudioSynth
// ---------------------------------------------------------------------------

pub struct AudioSynth {
    sample_rate: u32,
    pings: Vec<Ping>,
    channels: [ChannelState; 3],

    /// Whether we've seen at least one frame (skip first to avoid loud spike)
    has_prev_frame: bool,

    /// Underwater delay line
    delay_buffer: [f32; DELAY_BUFFER_SIZE],
    delay_pos: usize,

    /// Low-pass filter state (stereo)
    lp_state_l: f32,
    lp_state_r: f32,

    /// Tremolo phase
    tremolo_phase: f32,

    /// LCG for variation
    noise_seed: u32,

    /// Smoothed activity for debug display
    smoothed_activity: f32,
}

impl AudioSynth {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            pings: Vec::with_capacity(MAX_PINGS),
            channels: [
                ChannelState {
                    cooldown: 0,
                    prev_variance: 0.0,
                    base_freq: 100.0,
                },
                ChannelState {
                    cooldown: 0,
                    prev_variance: 0.0,
                    base_freq: 300.0,
                },
                ChannelState {
                    cooldown: 0,
                    prev_variance: 0.0,
                    base_freq: 750.0,
                },
            ],
            has_prev_frame: false,
            delay_buffer: [0.0; DELAY_BUFFER_SIZE],
            delay_pos: 0,
            lp_state_l: 0.0,
            lp_state_r: 0.0,
            tremolo_phase: 0.0,
            noise_seed: 0xC0FFEE,
            smoothed_activity: 0.0,
        }
    }

    pub fn activity_level(&self) -> f32 {
        self.smoothed_activity
    }

    // --- Random ---

    fn rand(&mut self) -> f32 {
        self.noise_seed = self
            .noise_seed
            .wrapping_mul(1_103_515_245)
            .wrapping_add(12_345);
        (self.noise_seed as f32) / (u32::MAX as f32)
    }

    fn rand_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.rand() * (max - min)
    }

    // --- Feature extraction ---

    pub fn extract_features(&mut self, render_rgb: &[u8], grid_size: usize) -> AudioFeatures {
        let n = grid_size * grid_size;
        let n_f = n as f64;

        let mut sums = [0.0f64; 3];
        let mut sum_sqs = [0.0f64; 3];

        for i in 0..n {
            let base = i * 3;
            for ch in 0..3 {
                let v = render_rgb[base + ch] as f64 / 255.0;
                sums[ch] += v;
                sum_sqs[ch] += v * v;
            }
        }

        let mut means = [0.0f32; 3];
        let mut variances = [0.0f32; 3];
        for ch in 0..3 {
            means[ch] = (sums[ch] / n_f) as f32;
            let mean_sq = (sum_sqs[ch] / n_f) as f32;
            variances[ch] = (mean_sq - means[ch] * means[ch]).max(0.0);
        }

        AudioFeatures {
            channel_means: means,
            channel_variances: variances,
        }
    }

    // --- Ping triggering ---

    fn trigger_pings(&mut self, features: &AudioFeatures) {
        // Skip the first frame — prev_variance starts at 0, so the initial
        // state would produce a massive spike.
        if !self.has_prev_frame {
            for ch in 0..3 {
                self.channels[ch].prev_variance = features.channel_variances[ch];
            }
            self.has_prev_frame = true;
            return;
        }

        let mut total_change = 0.0f32;

        for ch in 0..3 {
            if self.channels[ch].cooldown > 0 {
                self.channels[ch].cooldown -= 1;
            }

            let change = (features.channel_variances[ch] - self.channels[ch].prev_variance).abs();
            total_change += change;

            self.channels[ch].prev_variance = features.channel_variances[ch];

            if self.channels[ch].cooldown > 0 || change < VAR_CHANGE_THRESHOLD {
                continue;
            }

            self.fire_ping(ch, change, features);
        }

        self.smoothed_activity = self.smoothed_activity * 0.85 + total_change * 0.15;
    }

    fn fire_ping(&mut self, ch: usize, change: f32, features: &AudioFeatures) {
        // Drop quietest if polyphony full
        if self.pings.len() >= MAX_PINGS {
            if let Some(pos) = self
                .pings
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.amplitude.partial_cmp(&b.amplitude).unwrap())
                .map(|(i, _)| i)
            {
                self.pings.remove(pos);
            } else {
                return;
            }
        }

        let base = self.channels[ch].base_freq;
        let brightness = features.channel_means[ch];

        // Frequency sweep: start high, sweep down (sonar-like).
        // Brighter channels sweep from higher frequencies.
        let center = base * (0.8 + brightness * 0.5);
        let sweep_up = self.rand() > 0.5;
        let (freq_start, freq_end) = if sweep_up {
            (center * 0.6, center * 1.4)
        } else {
            (center * 1.5, center * 0.5)
        };

        // Duration: bigger change → longer ping
        let change_norm = (change / 0.00005).min(1.0);
        let duration = PING_MIN_SAMPLES
            + ((PING_MAX_SAMPLES - PING_MIN_SAMPLES) as f32 * change_norm) as usize;

        // Amplitude
        let amplitude = (change * 23400.0).min(0.95).max(0.10);

        // Stereo pan
        let base_pan = [0.15, 0.5, 0.85][ch];
        let pan = (base_pan + self.rand_range(-0.2, 0.2)).clamp(0.0, 1.0);

        // Cooldown: bigger change → shorter cooldown
        let cooldown =
            COOLDOWN_MIN + ((COOLDOWN_MAX - COOLDOWN_MIN) as f32 * (1.0 - change_norm)) as usize;
        self.channels[ch].cooldown = cooldown;

        let phase = self.rand_range(0.0, TAU);
        self.pings.push(Ping {
            phase,
            freq_start,
            freq_end,
            remaining: duration,
            total_duration: duration,
            amplitude,
            pan,
        });
    }

    // --- Audio frame generation ---

    pub fn generate_frame(&mut self, features: &AudioFeatures, num_samples: usize) -> Vec<i16> {
        self.trigger_pings(features);

        let sr = self.sample_rate as f32;
        let lp_alpha = 1.0 - (-TAU * LP_CUTOFF / sr).exp();
        let mut output = Vec::with_capacity(num_samples * 2);

        for i in 0..num_samples {
            let t = i as f32;

            // --- Tremolo LFO (alien pulsating) ---
            let tremolo = (TAU * TREMOLO_RATE * t / sr + self.tremolo_phase).sin() * 0.15 + 0.85;

            // --- Mix all active pings ---
            let mut left = 0.0f32;
            let mut right = 0.0f32;

            for ping in &mut self.pings {
                if !ping.is_alive() {
                    continue;
                }

                // Envelope: fast attack, exponential decay
                let progress = ping.remaining as f32 / ping.total_duration as f32;
                let env = if progress > 0.93 {
                    (1.0 - progress) * 14.0
                } else {
                    (progress / 0.93).powf(1.6)
                };
                let env = env.clamp(0.0, 1.0);

                // Frequency-swept sine
                let freq = ping.current_freq();
                // Phase increment for this sample (variable frequency)
                let phase_inc = TAU * freq / sr;
                let tone = ping.phase.sin();
                ping.phase += phase_inc;
                if ping.phase >= TAU {
                    ping.phase -= TAU;
                }

                // Add subtle 2nd harmonic that decays faster
                let harm2 = (ping.phase * 2.0).sin() * 0.15 * env;

                let sample = (tone + harm2) * env * ping.amplitude * tremolo;

                left += sample * (1.0 - ping.pan);
                right += sample * ping.pan;

                ping.remaining = ping.remaining.saturating_sub(1);
            }

            self.pings.retain(|p| p.is_alive());

            // --- Low-pass filter ---
            self.lp_state_l += lp_alpha * (left - self.lp_state_l);
            self.lp_state_r += lp_alpha * (right - self.lp_state_r);
            let filtered_l = self.lp_state_l;
            let filtered_r = self.lp_state_r;

            // --- Feedback delay ---
            let delay_read =
                (self.delay_pos + DELAY_BUFFER_SIZE - DELAY_TIME_SAMPLES) % DELAY_BUFFER_SIZE;
            let echo_l = self.delay_buffer[delay_read];
            let echo_r = self.delay_buffer[(delay_read + 1) % DELAY_BUFFER_SIZE];

            let wet_l = filtered_l + echo_l * DELAY_WET;
            let wet_r = filtered_r + echo_r * DELAY_WET;

            self.delay_buffer[self.delay_pos] = filtered_l + echo_l * DELAY_FEEDBACK;
            self.delay_buffer[(self.delay_pos + 1) % DELAY_BUFFER_SIZE] =
                filtered_r + echo_r * DELAY_FEEDBACK;
            self.delay_pos = (self.delay_pos + 2) % DELAY_BUFFER_SIZE;

            // Linear pre-gain with tanh as safety limiter
            let out_l = (wet_l * 5.0).tanh();
            let out_r = (wet_r * 5.0).tanh();

            let l = (out_l * 32767.0).clamp(-32768.0, 32767.0) as i16;
            let r = (out_r * 32767.0).clamp(-32768.0, 32767.0) as i16;

            output.push(l);
            output.push(r);
        }

        // Advance tremolo phase
        self.tremolo_phase += TAU * TREMOLO_RATE * num_samples as f32 / sr;
        self.tremolo_phase %= TAU;

        output
    }
}

// ---------------------------------------------------------------------------
// WAV writer
// ---------------------------------------------------------------------------

pub fn write_wav(
    path: &str,
    samples: &[i16],
    sample_rate: u32,
    num_channels: u16,
) -> std::io::Result<()> {
    use std::io::Write;

    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    let byte_rate = sample_rate * num_channels as u32 * 2;
    file.write_all(&byte_rate.to_le_bytes())?;
    let block_align = num_channels * 2;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;

    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    for &sample in samples {
        file.write_all(&sample.to_le_bytes())?;
    }

    Ok(())
}

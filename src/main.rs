use ggez::event::{self, EventHandler};
use ggez::graphics::{Canvas, Color, DrawMode, DrawParam, Mesh, MeshBuilder, Rect, Text};
use ggez::input::keyboard::{KeyCode, KeyInput};
use ggez::{Context, ContextBuilder, GameResult};
use lenia_ca::lenias::ExpandedLenia;
use lenia_ca::{growth_functions, kernels, Simulator};
use ndarray::Array2;
use rand::Rng;

const RESOLUTION: f32 = 1400.0;
const FPS: u32 = 60;
const GRID_SIZE: usize = 300;
const NUM_CHANNELS: usize = 2;

struct LeniaWorld {
    game: Simulator<ExpandedLenia>,
    iteration: u64,
    paused: bool,
    show_channel: usize,
    shape: usize,
    screen_size: f32,
    stats: PopulationStats,
}

#[derive(Default)]
struct PopulationStats {
    apex: f64,
    predator: f64,
    prey: f64,
    plant: f64,
}

impl LeniaWorld {
    fn new() -> Self {
        let mut game = Simulator::<ExpandedLenia>::new(&[GRID_SIZE, GRID_SIZE]);

        // Setup 2 channels and 2 convolution channels - simple predator-prey
        game.set_channels(NUM_CHANNELS);
        game.set_convolution_channels(2);

        // === Channel 0: Orbium Gliders (Classic Lenia creatures) ===
        game.set_convolution_channel_source(0, 0);
        game.set_kernel(kernels::gaussian_donut_2d(13, 1.0 / 6.7), 0);
        game.set_growth_function(growth_functions::standard_lenia, vec![0.15, 0.017], 0);

        // === Channel 1: Different Species with Interaction ===
        game.set_convolution_channel_source(1, 1);
        game.set_kernel(kernels::gaussian_donut_2d(13, 1.0 / 6.0), 1);
        game.set_growth_function(growth_functions::standard_lenia, vec![0.135, 0.015], 1);

        // Interaction weights - mostly independent with slight interaction
        game.set_weights(0, &[1.0, 0.0]);
        game.set_weights(1, &[0.0, 1.0]);

        // Classic Lenia time step
        game.set_dt(0.1);

        let mut world = LeniaWorld {
            game,
            iteration: 0,
            paused: false,
            show_channel: 4, // Start with composite view
            shape: GRID_SIZE,
            screen_size: RESOLUTION,
            stats: PopulationStats::default(),
        };

        world.initialize_world();
        world
    }

    fn initialize_world(&mut self) {
        let mut rng = rand::thread_rng();

        // Channel 0: Classic Orbium gliders
        let mut ch0 = Array2::<f64>::zeros([self.shape, self.shape]);
        for _ in 0..8 {
            let x = rng.gen_range(50..self.shape - 50);
            let y = rng.gen_range(50..self.shape - 50);
            add_orbium(&mut ch0, x, y);
        }
        self.game.fill_channel(&ch0.into_dyn(), 0);

        // Channel 1: Slightly different species
        let mut ch1 = Array2::<f64>::zeros([self.shape, self.shape]);
        for _ in 0..6 {
            let x = rng.gen_range(50..self.shape - 50);
            let y = rng.gen_range(50..self.shape - 50);
            add_orbium_variant(&mut ch1, x, y);
        }
        self.game.fill_channel(&ch1.into_dyn(), 1);
    }

    fn update_stats(&mut self) {
        let ch0 = self.game.get_channel_as_ref(0);
        let ch1 = self.game.get_channel_as_ref(1);

        self.stats.apex = ch0.iter().sum();
        self.stats.predator = ch1.iter().sum();
        self.stats.prey = 0.0;
        self.stats.plant = 0.0;
    }
}

// Classic Orbium pattern from Lenia paper
fn add_orbium(array: &mut Array2<f64>, cx: usize, cy: usize) {
    let shape = [array.shape()[0], array.shape()[1]];
    let radius = 20;

    let i_min = (cx as i32 - radius).max(0) as usize;
    let i_max = (cx as usize + radius as usize).min(shape[0]);
    let j_min = (cy as i32 - radius).max(0) as usize;
    let j_max = (cy as usize + radius as usize).min(shape[1]);

    for i in i_min..i_max {
        for j in j_min..j_max {
            let dx = i as f64 - cx as f64;
            let dy = j as f64 - cy as f64;
            let r = (dx * dx + dy * dy).sqrt() / 13.0;

            if r < 1.0 {
                // Gaussian ring pattern
                let val = (-((r - 0.5) * (r - 0.5)) / (2.0 * 0.15 * 0.15)).exp();
                array[[i, j]] = (array[[i, j]] + val * 0.5).min(1.0);
            }
        }
    }
}

// Variant orbium with slightly different parameters
fn add_orbium_variant(array: &mut Array2<f64>, cx: usize, cy: usize) {
    let shape = [array.shape()[0], array.shape()[1]];
    let radius = 18;

    let i_min = (cx as i32 - radius).max(0) as usize;
    let i_max = (cx as usize + radius as usize).min(shape[0]);
    let j_min = (cy as i32 - radius).max(0) as usize;
    let j_max = (cy as usize + radius as usize).min(shape[1]);

    for i in i_min..i_max {
        for j in j_min..j_max {
            let dx = i as f64 - cx as f64;
            let dy = j as f64 - cy as f64;
            let r = (dx * dx + dy * dy).sqrt() / 11.0;

            if r < 1.2 {
                let val = (-((r - 0.55) * (r - 0.55)) / (2.0 * 0.18 * 0.18)).exp();
                array[[i, j]] = (array[[i, j]] + val * 0.45).min(1.0);
            }
        }
    }
}

impl EventHandler for LeniaWorld {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        while ctx.time.check_update_time(FPS) {
            if !self.paused {
                self.game.iterate();
                self.iteration += 1;

                // Update stats every 10 iterations for performance
                if self.iteration % 10 == 0 {
                    self.update_stats();
                }
            }
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, Color::new(0.01, 0.01, 0.02, 1.0));
        let mut builder = MeshBuilder::new();
        let cell_size = self.screen_size / self.shape as f32;

        match self.show_channel {
            0..=1 => {
                // Individual channel view with species-specific colors
                let cells = self.game.get_channel_as_ref(self.show_channel);
                let color_base = match self.show_channel {
                    0 => Color::new(0.0, 1.0, 0.5, 1.0), // Cyan for species 1
                    _ => Color::new(1.0, 0.5, 0.0, 1.0), // Orange for species 2
                };

                cells
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x > 0.015)
                    .for_each(|(i, &x)| {
                        let pos_x = (i % self.shape) as f32;
                        let pos_y = (i / self.shape) as f32;
                        let intensity = x.min(1.0).powf(0.65) as f32;
                        let color = Color::new(
                            color_base.r * intensity,
                            color_base.g * intensity,
                            color_base.b * intensity,
                            intensity.powf(0.5) * 0.85,
                        );
                        let rect =
                            Rect::new(pos_x * cell_size, pos_y * cell_size, cell_size, cell_size);
                        builder.rectangle(DrawMode::fill(), rect, color).unwrap();
                    });
            }
            _ => {
                // Composite view
                let ch0 = self.game.get_channel_as_ref(0);
                let ch1 = self.game.get_channel_as_ref(1);

                for i in 0..self.shape {
                    for j in 0..self.shape {
                        let v0 = ch0[[i, j]].min(1.0) as f32;
                        let v1 = ch1[[i, j]].min(1.0) as f32;

                        // Color mixing: Cyan + Orange
                        let red = v1 * 1.0 + v0 * 0.1;
                        let green = v0 * 1.0 + v1 * 0.5;
                        let blue = v0 * 0.5 + v1 * 0.1;

                        let total = (red + green + blue) / 3.0;

                        if total > 0.025 {
                            let color = Color::new(
                                red.min(1.0),
                                green.min(1.0),
                                blue.min(1.0),
                                (total * 0.85).min(1.0),
                            );
                            let rect = Rect::new(
                                i as f32 * cell_size,
                                j as f32 * cell_size,
                                cell_size,
                                cell_size,
                            );
                            builder.rectangle(DrawMode::fill(), rect, color).unwrap();
                        }
                    }
                }
            }
        }

        let mesh = builder.build();
        let mesh = Mesh::from_data(ctx, mesh);
        canvas.draw(&mesh, DrawParam::default());

        // Draw comprehensive UI overlay
        let channel_name = match self.show_channel {
            0 => "Channel 0: Orbium Species 1 (Cyan) - Classic Gliders",
            1 => "Channel 1: Orbium Species 2 (Orange) - Variant Gliders",
            _ => "Composite View: Both Species",
        };

        let info = format!(
            "Iteration: {} | FPS: {:.0} | {}",
            self.iteration,
            ctx.time.fps(),
            channel_name
        );

        let stats = format!(
            "Population → Species 1: {:.0} | Species 2: {:.0}",
            self.stats.apex, self.stats.predator
        );

        let controls = "[Space]=Pause | [0-2]=Views | [R]=Reset | [Q/Esc]=Quit";

        let status = if self.paused { " [PAUSED]" } else { "" };

        draw_text(
            &mut canvas,
            &(info + status),
            12.0,
            12.0,
            19.0,
            Color::WHITE,
        );
        draw_text(
            &mut canvas,
            &stats,
            12.0,
            38.0,
            17.0,
            Color::new(0.7, 0.9, 1.0, 1.0),
        );
        draw_text(
            &mut canvas,
            controls,
            12.0,
            self.screen_size - 35.0,
            16.0,
            Color::new(0.6, 0.6, 0.6, 0.9),
        );

        canvas.finish(ctx)
    }

    fn key_down_event(&mut self, ctx: &mut Context, input: KeyInput, _repeat: bool) -> GameResult {
        if let Some(keycode) = input.keycode {
            match keycode {
                KeyCode::Space => self.paused = !self.paused,
                KeyCode::Key0 => self.show_channel = 0,
                KeyCode::Key1 => self.show_channel = 1,
                KeyCode::Key2 => self.show_channel = 2,
                KeyCode::Key2 => self.show_channel = 2,
                KeyCode::R => {
                    self.iteration = 0;
                    self.initialize_world();
                }
                KeyCode::Q | KeyCode::Escape => ctx.request_quit(),
                _ => {}
            }
        }
        Ok(())
    }
}

fn draw_text(canvas: &mut Canvas, text: &str, x: f32, y: f32, size: f32, color: Color) {
    let mut t = Text::new(text);
    t.set_scale(size);
    canvas.draw(&t, DrawParam::default().dest([x, y]).color(color));
}

fn main() -> GameResult {
    let world = LeniaWorld::new();

    let cb = ContextBuilder::new("Lenia Ecosystem Simulator", "Zoran Lazovic")
        .window_mode(ggez::conf::WindowMode::default().dimensions(RESOLUTION, RESOLUTION));

    let (ctx, event_loop) = cb.build()?;
    ctx.gfx
        .set_window_title("Lenia: Orbium Gliders - Self-Organizing Life");

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║       🌊 Lenia: Orbium Glider Simulator 🌊             ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  2-Channel System: Two Species of Moving Creatures     ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Controls:                                              ║");
    println!("║    [Space]     Pause/Resume simulation                 ║");
    println!("║    [0-1]       View individual species                 ║");
    println!("║    [2]         Composite view (both species)           ║");
    println!("║    [R]         Reset with new random creatures         ║");
    println!("║    [Q/Esc]     Quit                                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Watch for:                                             ║");
    println!("║    • Gliding orbium creatures                          ║");
    println!("║    • Self-organizing patterns                          ║");
    println!("║    • Collisions and interactions                       ║");
    println!("║    • Emergence of new creatures                        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    event::run(ctx, event_loop, world)
}

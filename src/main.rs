use glow::*;
use glutin::event::VirtualKeyCode;
use std::collections::HashSet;
use std::time::Instant;
use glutin::event::Event;
use glutin::event::WindowEvent;
use glutin::event::MouseButton;
use glutin::event::ElementState;
use cpal::Stream;
use cpal::traits::*;
use ringbuf::*;
use core::f32::consts::PI;


// extreme dopamine mode: scores queue up and get loaded in
// or if they replenish can you go infinite
// maybe with time constraint

// make the squares trippy fragment shader programs

// ====================
// Math
// ====================
pub fn khash(mut state: usize) -> usize {
    state = (state ^ 2747636419).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

#[derive(Clone, Copy, Debug)]
pub struct V2 {
    x: f32,
    y: f32,
}
fn v2(x: f32, y: f32) -> V2 { V2 { x, y } }
#[derive(Clone, Copy, Debug)]
pub struct V3 {
    x: f32,
    y: f32,
    z: f32,
}
fn v3(x: f32, y: f32, z: f32) -> V3 { V3 { x, y, z } }
#[derive(Clone, Copy, Debug)]
pub struct V4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}
fn v4(x: f32, y: f32, z: f32, w: f32) -> V4 { V4 { x, y, z, w } }

impl V2 {
    pub fn dot(&self, other: V2) -> f32 {
        self.x*other.x + self.y * other.y
    }
}
impl V3 {
    pub fn dot(&self, other: V3) -> f32 {
        self.x*other.x + self.y * other.y + self.z*other.z
    }
}
impl V4 {
    pub fn dot(&self, other: V4) -> f32 {
        self.x*other.x + self.y * other.y + self.z*other.z + self.w*other.w
    }
    pub fn tl(&self) -> V2 {v2(self.x, self.y)}
    pub fn br(&self) -> V2 {v2(self.x + self.z, self.y + self.w)}
    pub fn tr(&self) -> V2 {v2(self.x + self.z, self.y)}
    pub fn bl(&self) -> V2 {v2(self.x, self.y + self.w)}
    pub fn grid_child(&self, i: usize, j: usize, w: usize, h: usize) -> V4 {
        let cw = self.z / w as f32;
        let ch = self.w / h as f32;
        v4(self.x + cw * i as f32, self.y + ch * j as f32, cw, ch)
    }
    pub fn hsv_to_rgb(&self) -> V4 {
        let v = self.z;
        let hh = (self.x % 360.0) / 60.0;
        let i = hh.floor() as i32;
        let ff = hh - i as f32;
        let p = self.z * (1.0 - self.y);
        let q = self.z * (1.0 - self.y * ff);
        let t = self.z * (1.0 - self.y * (1.0 - ff));
        match i {
            0 => v4(v, t, p, self.w),
            1 => v4(q, v, p, self.w),
            2 => v4(p, v, t, self.w),
            3 => v4(p, q, v, self.w),
            4 => v4(t, p, v, self.w),
            5 => v4(v, p, q, self.w),
            _ => panic!("unreachable"),
        }
    }
    fn contains(&self, p: V2) -> bool {
        !(p.x < self.x || p.x > self.x + self.z || p.y < self.y || p.y > self.y + self.w)
    }
    fn point_within(&self, p: V2) -> V2 {
        v2(p.x*self.z+self.x, p.y*self.w+self.y)
    }
    fn point_without(&self, p: V2) -> V2 {
        v2((p.x - self.x) / self.z, (p.y - self.y) / self.w)
    }
}


// ====================
// Canvas
// ====================
pub struct CTCanvas {
    a: f32,
    buf: Vec<u8>,
}

impl CTCanvas {
    pub fn new(a: f32) -> CTCanvas {
        CTCanvas {
            a,
            buf: Vec::new(),
        }
    }

    fn put_u32(&mut self, x: u32) {
        for b in x.to_le_bytes() {
            self.buf.push(b);
        }
    }

    fn put_float(&mut self, x: f32) {
        for b in x.to_le_bytes() {
            self.buf.push(b);
        }
    }

    pub fn put_triangle(&mut self, p1: V2, uv1: V2, p2: V2, uv2: V2, p3: V2, uv3: V2, depth: f32, colour: V4, mode: u32) {
        self.put_float(p1.x/self.a);
        self.put_float(p1.y);
        self.put_float(depth);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);
        self.put_float(uv1.x);
        self.put_float(uv1.y);
        self.put_u32(mode);
        
        self.put_float(p2.x/self.a);
        self.put_float(p2.y);
        self.put_float(depth);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);
        self.put_float(uv2.x);
        self.put_float(uv2.y);
        self.put_u32(mode);
        
        self.put_float(p3.x/self.a);
        self.put_float(p3.y);
        self.put_float(depth);
        self.put_float(colour.x);
        self.put_float(colour.y);
        self.put_float(colour.z);
        self.put_float(colour.w);
        self.put_float(uv3.x);
        self.put_float(uv3.y);
        self.put_u32(mode);
    }

    pub fn put_rect(&mut self, r: V4, r_uv: V4, depth: f32, colour: V4, mode: u32) {
        self.put_triangle(r.tl(), r_uv.tl(), r.tr(), r_uv.tr(), r.bl(), r_uv.bl(), depth, colour, mode);
        self.put_triangle(r.bl(), r_uv.bl(), r.tr(), r_uv.tr(), r.br(), r_uv.br(), depth, colour, mode);
    }

    pub fn put_glyph(&mut self, c: char, r: V4, depth: f32, colour: V4) {
        let clip_fn = |mut c: u8| {
            if c >= 'a' as u8 && c <= 'z' as u8 {
                c -= 'a' as u8 - 'A' as u8;
            }
            if c >= '+' as u8 && c <= '_' as u8 {
                let x = c - '+' as u8;
                let w = '_' as u8 - '+' as u8 + 1; // maybe +1
                Some(v4(0.0, 0.0, 1.0, 1.0).grid_child(x as usize, 0, w as usize, 1))
            } else {
                None
            }
        };
        if let Some(r_uv) = clip_fn(c as u8) {
            self.put_rect(r, r_uv, depth, colour, 1);
        }
    }

    pub fn put_string_left(&mut self, s: &str, mut x: f32, y: f32, cw: f32, ch: f32, depth: f32, colour: V4) {
        for c in s.chars() {
            self.put_glyph(c, v4(x, y, cw, ch), depth, colour);
            x += cw;
        }
    }
    pub fn put_string_centered(&mut self, s: &str, mut x: f32, mut y: f32, cw: f32, ch: f32, depth: f32, colour: V4) {
        let w = s.len() as f32 * cw;
        x -= w/2.0;
        // y -= ch/2.0;
        for c in s.chars() {
            self.put_glyph(c, v4(x, y, cw, ch), depth, colour);
            x += cw;
        }
    }
}

// ====================
// Audio stuff
// ====================
fn sample_next(o: &mut SampleRequestOptions) -> f32 {
    let mut acc = 0.0;
    let mut idx = o.sounds.len();
    loop {
        if idx == 0 {
            break;
        }
        idx -= 1;

        o.sounds[idx].elapsed += 1.0/44100.0;
        o.sounds[idx].remaining -= 1.0/44100.0;

        let t = o.sounds[idx].elapsed;

        if o.sounds[idx].remaining < 0.0 {
            o.sounds.swap_remove(idx);
            continue;
        }
        if o.sounds[idx].id == 0 {
            o.sounds[idx].magnitude *= 0.999;

            let f = o.sounds[idx].frequency;
            let f_trans = f*3.0;

            let t_trans = 1.0/(2.0*PI*f_trans);

            if o.sounds[idx].elapsed < t_trans {
                o.sounds[idx].phase += f_trans*2.0*PI*1.0/o.sample_rate;
            } else {
                o.sounds[idx].phase += f*2.0*PI*1.0/o.sample_rate;
            }
            // o.sounds[idx].phase += f*2.0*PI*1.0/o.sample_rate;

            //o.sounds[idx].phase = o.sounds[idx].phase % 2.0*PI; // this sounds really good lol

            acc += (o.sounds[idx].phase).sin() * o.sounds[idx].magnitude
        } else if o.sounds[idx].id == 1 {
            o.sounds[idx].magnitude *= 0.9999;
            // o.sounds[idx].frequency *= 0.99999;
            o.sounds[idx].frequency *= 0.99999;
            // fast freq drop will do as well, sounds gud actually

            o.sounds[idx].phase += o.sounds[idx].frequency*2.0*PI*1.0/o.sample_rate;

            acc += (o.sounds[idx].phase).sin() * o.sounds[idx].magnitude
        }
    }
    acc
}

#[derive(Debug)]
pub struct Sound {
    id: usize,
    birthtime: f32,
    elapsed: f32,
    remaining: f32,
    magnitude: f32,
    frequency: f32,
    phase: f32,
}

pub struct SampleRequestOptions {
    pub sample_rate: f32,
    pub nchannels: usize,
    pub channel: Consumer<Sound>,
    pub sounds: Vec<Sound>,
}

pub fn stream_setup_for<F>(on_sample: F, channel: Consumer<Sound>) -> Result<cpal::Stream, anyhow::Error>
where
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static + Copy,
{
    let (_host, device, config) = host_device_setup()?;

    match config.sample_format() {
        cpal::SampleFormat::F32 => stream_make::<f32, _>(&device, &config.into(), on_sample, channel),
        cpal::SampleFormat::I16 => stream_make::<i16, _>(&device, &config.into(), on_sample, channel),
        cpal::SampleFormat::U16 => stream_make::<u16, _>(&device, &config.into(), on_sample, channel),
    }
}

pub fn host_device_setup(
) -> Result<(cpal::Host, cpal::Device, cpal::SupportedStreamConfig), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::Error::msg("Default output device is not available"))?;
    println!("Output device : {}", device.name()?);

    let config = device.default_output_config()?;
    println!("Default output config : {:?}", config);

    Ok((host, device, config))
}


pub fn stream_make<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    on_sample: F,
    channel: Consumer<Sound>,
) -> Result<cpal::Stream, anyhow::Error>
where
    T: cpal::Sample,
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static + Copy,
{
    let sample_rate = config.sample_rate.0 as f32;
    let nchannels = config.channels as usize;
    let mut request = SampleRequestOptions {
        sample_rate,
        nchannels,
        sounds: vec![],
        channel,
    };
    let err_fn = |err| eprintln!("Error building output sound stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            on_window(output, &mut request, on_sample)
        },
        err_fn,
    )?;

    Ok(stream)
}

fn on_window<T, F>(output: &mut [T], request: &mut SampleRequestOptions, mut on_sample: F)
where
    T: cpal::Sample,
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static,
{
    if let Some(sc) = request.channel.pop() {
        request.sounds.push(sc);
    }
    for frame in output.chunks_mut(request.nchannels) {
        let value: T = cpal::Sample::from::<f32>(&on_sample(request));
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}


fn main() {
    unsafe {
        let xres = 1600i32;
        let yres = 1600i32;

        // ====================
        // Sound Init
        // ====================
        let rb = RingBuffer::<Sound>::new(100);
        let (mut prod, mut cons) = rb.split();
        let stream = stream_setup_for(sample_next, cons).expect("no can make stream");
        stream.play().expect("no can play stream");

        let event_loop = glutin::event_loop::EventLoop::new();
        let window_builder = glutin::window::WindowBuilder::new()
                .with_title("shadytoy")
                .with_inner_size(glutin::dpi::PhysicalSize::new(xres, yres));

        let window = glutin::ContextBuilder::new()
                .with_vsync(true)
                .build_windowed(window_builder, &event_loop)
                .unwrap()
                .make_current()
                .unwrap();


        // ====================
        // GL init
        // ====================
        let gl = glow::Context::from_loader_function(|s| window.get_proc_address(s) as *const _);
        gl.enable(DEPTH_TEST);
        // gl.enable(CULL_FACE);
        gl.blend_func(SRC_ALPHA, ONE_MINUS_SRC_ALPHA);
        gl.enable(BLEND);
        // gl.debug_message_callback(|a, b, c, d, msg| {
        //     println!("{} {} {} {} msg: {}", a, b, c, d, msg);
        // });

        let vbo = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

        let vao = gl.create_vertex_array().unwrap();
        gl.bind_vertex_array(Some(vao));
        
        gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 4*4 + 4*3 + 4*2 + 4, 0);
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(1, 4, glow::FLOAT, false, 4*4 + 4*3 + 4*2 + 4, 4*3);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, 4*4 + 4*3 + 4*2 + 4, 4*3 + 4*4);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_i32(3, 1, glow::UNSIGNED_INT, 4*4 + 4*3 + 4*2 + 4, 4*3 + 4*4 + 4*2);
        gl.enable_vertex_attrib_array(3);


        // Shader
        let program = gl.create_program().expect("Cannot create program");
    
        let vs = gl.create_shader(glow::VERTEX_SHADER).expect("cannot create vertex shader");
        gl.shader_source(vs, include_str!("shader.vert"));
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            panic!("{}", gl.get_shader_info_log(vs));
        }
        gl.attach_shader(program, vs);

        let fs = gl.create_shader(glow::FRAGMENT_SHADER).expect("cannot create fragment shader");
        gl.shader_source(fs, include_str!("shader.frag"));
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            panic!("{}", gl.get_shader_info_log(fs));
        }
        gl.attach_shader(program, fs);

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("{}", gl.get_program_info_log(program));
        }
        gl.detach_shader(program, fs);
        gl.delete_shader(fs);
        gl.detach_shader(program, vs);
        gl.delete_shader(vs);

        let png_bytes = include_bytes!("../font.png").as_ref();
        let decoder = png::Decoder::new(png_bytes);
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let bytes = &buf[..info.buffer_size()];

        let texture = gl.create_texture().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D, 
            0, 
            glow::RGBA as i32, 
            info.width as i32, info.height as i32, 
            0, 
            RGBA, 
            glow::UNSIGNED_BYTE, 
            Some(bytes)
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        gl.generate_mipmap(glow::TEXTURE_2D);





        // ====================
        // Simulation
        // ====================
        let mut score = 0;
        let mut display_score = 0;
        let bw = 10;
        let bh = 10;
        let num_colours = 5;
        let mut board = vec![];
        let mut seed = 69;
        let mut t = 0.0;
        let mut t_last = Instant::now();
        for i in 0..bw*bh {
            board.push(Some(khash(seed + i * 12312389) % num_colours));
        }
        let mut mouse_pos = v2(0., 0.);


        event_loop.run(move |event, _, _| {
            match event {
                Event::LoopDestroyed |
                Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => {
                    std::process::exit(0);
                }

                Event::WindowEvent {event, .. } => {
                    match event {
                        WindowEvent::CursorMoved {position, .. } => {
                            mouse_pos.x = position.x as f32 / xres as f32;
                            mouse_pos.y = position.y as f32 / yres as f32;
                        },
                        WindowEvent::MouseInput {state: ElementState::Pressed, button: MouseButton::Left, ..} => {
                        },
                        WindowEvent::KeyboardInput {input, ..} => {
                            match input {
                                glutin::event::KeyboardInput {virtual_keycode: Some(code), state: ElementState::Released, ..} => {
                                    match code {
                                        VirtualKeyCode::R => {
                                            seed += 1;
                                            board = vec![];
                                            for i in 0..bw*bh {
                                                board.push(Some(khash(seed + i * 12312389) % num_colours));
                                            }
                                        },
                                        _ => {},
                                    }
                                },
                                _ => {},
                            }
                        },
                        _ => {},
                    }
                }

                Event::MainEventsCleared => {
                    let t_now = Instant::now();
                    let dt = (t_now - t_last).as_secs_f64();
                    t += dt;
                    t_last = t_now;

                    let mut canvas = CTCanvas::new(xres as f32 / yres as f32);

                    // draw background
                    canvas.put_rect(v4(0., 0., 1., 1.), v4(0., 0., 1., 1.), 1.3, v4(1., 1., 1., 1.,), 1002);
                    // canvas.put_rect(v4(0., 0., 1., 1.), v4(0., 0., 1., 1.), 1.3, v4(1., 1., 1., 1.,), 1000);

                    gl.uniform_1_f32(gl.get_uniform_location(program, "time").as_ref(), t as f32);


                    gl.clear_color(0.0, 0.0, 0.0, 1.0);
                    gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT); 
                    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                    gl.use_program(Some(program));
                    gl.bind_vertex_array(Some(vao));
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &canvas.buf, glow::DYNAMIC_DRAW);
                    let vert_count = canvas.buf.len() / (9*4);
                    gl.draw_arrays(glow::TRIANGLES, 0, vert_count as i32);
                    window.swap_buffers().unwrap();
                }

                _ => {},
            }
        });

    }
}
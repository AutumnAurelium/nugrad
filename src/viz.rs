// In viz.rs

use raylib::{prelude::*, color::Color};
use std::{collections::HashMap, rc::Rc, sync::{Arc, RwLock}};
use crate::{read_lock, value::*};

const NODE_RADIUS: f32 = 20.0;
const HORIZONTAL_SPACING: f32 = 60.0;
const VERTICAL_SPACING: f32 = 40.0;
const PADDING: f32 = 60.0;  // Added padding constant

pub fn visualize(output: ValueLock) {
    let (mut rl, thread) = raylib::init()
        .size(1024, 768)
        .title("NuGrad Visualization")
        .build();
    
    let mut positions = HashMap::new();
    let depth = read_lock!(output).depth();
    let width = calculate_width(&output);

    calculate_positions(&output, 0, 0, width, &mut positions);

    // Adjust positions to add padding
    let (min_x, min_y, max_x, max_y) = get_bounds(&positions);
    let scale_x = (rl.get_screen_width() as f32 - 2.0 * PADDING) / (max_x - min_x);
    let scale_y = (rl.get_screen_height() as f32 - 2.0 * PADDING) / (max_y - min_y);
    let scale = scale_x.min(scale_y) * 0.9;  // Added a factor to leave some extra space
    for (_, pos) in positions.iter_mut() {
        pos.0 = (pos.0 - min_x) * scale + PADDING;
        pos.1 = (pos.1 - min_y) * scale + PADDING;
    }

    rl.set_target_fps(30);

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        
        d.clear_background(Color::BLACK);
        draw_graph(&mut d, &output, &positions, 0);
    }
}

fn calculate_width(node: &ValueLock) -> usize {
    let value = read_lock!(node);
    if value.parents().is_empty() {
        1
    } else {
        value.parents().iter().map(|p| calculate_width(p)).sum()
    }
}

fn calculate_positions(node: &ValueLock, depth: usize, index: usize, width: usize, positions: &mut HashMap<*const RwLock<Value>, (f32, f32)>) {
    let x = (index as f32 + 0.5) * HORIZONTAL_SPACING * width as f32 / calculate_width(node) as f32;
    let y = depth as f32 * VERTICAL_SPACING;

    positions.insert(Arc::as_ptr(node), (x, y));

    let value = read_lock!(node);
    let mut current_index = index;
    for parent in value.parents() {
        let parent_width = calculate_width(parent);
        calculate_positions(parent, depth + 1, current_index, parent_width, positions);
        current_index += parent_width;
    }
}

fn get_bounds(positions: &HashMap<*const RwLock<Value>, (f32, f32)>) -> (f32, f32, f32, f32) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for &(x, y) in positions.values() {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    (min_x, min_y, max_x, max_y)
}

fn draw_graph(d: &mut RaylibDrawHandle, node: &ValueLock, positions: &HashMap<*const RwLock<Value>, (f32, f32)>, depth: usize) {
    if depth >= 10 {
        return;
    }
    let value = read_lock!(node);
    let (x, y) = *positions.get(&Arc::as_ptr(node)).unwrap();

    // Draw edges
    for parent in value.parents() {
        let (parent_x, parent_y) = *positions.get(&Arc::as_ptr(parent)).unwrap();
        d.draw_line(x as i32, y as i32, parent_x as i32, parent_y as i32, Color::GRAY);
    }

    // Draw node
    let color = match value.operator() {
        Operator::Variable => {
            if value.needs_grad() {
                Color::ORANGE
            } else {
                Color::GREEN
            }
        },
        _ => Color::BLUE,
    };
    d.draw_circle(x as i32, y as i32, NODE_RADIUS, color);

    // Draw label
    let label = format!("{}", value);
    let text_size = 10;
    let text_width = d.measure_text(&label, text_size);
    d.draw_text(&label, (x - text_width as f32 / 2.0) as i32, (y - text_size as f32 / 2.0) as i32, text_size, Color::WHITE);

    // Recursively draw parents
    for parent in value.parents() {
        draw_graph(d, parent, positions, depth+1);
    }
}


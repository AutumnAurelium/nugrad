use std::{fs::File, io::{BufRead, BufReader}, iter, result::IntoIter, time::Instant};

use nn::*;
use serde::{Deserialize, Serialize};
use serde_json::{de::IoRead, Deserializer};
use value::{as_values, sum, ParentVec, Value, ValueLock};
use viz::visualize;

pub mod value;
pub mod nn;
pub mod viz;

#[derive(Serialize, Deserialize, Debug)]
struct MnistSample {
    pub image: Vec<u8>,
    pub label: u8
}

/**
 * Loads at most `load_samples` samples from a JSONL file and returns their representation.
 */
fn load_mnist(filename: &str, load_samples: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    assert!(filename.ends_with(".jsonl"));

    let file = File::open(filename).expect(format!("Could not open {}", filename).as_str());
    let reader = BufReader::new(file);

    let mut data = vec![];

    for (i, result) in reader.lines().enumerate() {
        let line = result.expect("Failed to read file");
        let image: MnistSample = serde_json::from_str(line.as_str()).expect("Failed to parse line");

        let mut image_vals = vec![];
        for pix_byte in image.image {
            image_vals.push(pix_byte as f32 / 256.0);
        }

        let mut labels = vec![0.0; 10];
        labels[image.label as usize] = 1.0;
        data.push((image_vals, labels));

        if i % (load_samples / 5) == 0 {
            println!("{}/{}", i, load_samples);
        }

        if i >= load_samples {
            break;
        }
    }

    return data;
}


fn main() {
    let input_count = 784;
    let neuron_counts = vec![10];
    let batch_size = 1;
    let iterations = 500;

    let initial_lr = 0.1;
    let decay_rate = 0.0;
    let learning_rate = |it| initial_lr * f32::exp(-decay_rate * it as f32);

    let activation = value::Operator::ReLU;

    println!("Loading data...");
    let data: Vec<(Vec<f32>, Vec<f32>)> = load_mnist("data/mnist_handwritten_train.jsonl", 1000);
    println!("Done!");

    let params = mlp_parameters(input_count, neuron_counts.clone());
    let mut params_flat = vec![];

    for layer_params in params.iter() {
        for neuron_params in layer_params {
            for param in neuron_params {
                params_flat.push(param);
            }
        }
    }
    
    println!("starting training on {} parameters and {} samples", params_flat.len(), iterations*batch_size);
    for it in 0..iterations {
        let mut batch_losses = vec![];
        let mut inputs = vec![];
        let mut outputs = vec![];
        let mut targets = vec![];

        for net_i in 0..batch_size {
            let (data_in, data_out) = &data[it*batch_size + net_i];

            let mut net_inputs = vec![];
            for i in 0..data_in.len() {
                net_inputs.push(Value::new(data_in[i]));
            }

            let mut net_targets = vec![];
            for i in 0..data_out.len() {
                net_targets.push(Value::new(data_out[i]));
            }
            
            let net = mlp(net_inputs.clone(), neuron_counts.clone(), &params, activation);
            let loss = rmse(net.clone(), net_targets.clone());

            inputs.push(net_inputs);
            targets.push(net_targets);
            outputs.push(net.clone());
            batch_losses.push(loss.clone());
        }

        let loss = sum(batch_losses);

        // visualize(loss.clone());

        println!("== {}/{} ==", it+1, iterations);

        println!("output[0]: {:?}", outputs[0].iter().map(|o| read_lock!(o).value()).collect::<Vec<f32>>());
        println!("target[0]: {:?}", targets[0].iter().map(|o| read_lock!(o).value()).collect::<Vec<f32>>());

        let actual_loss = {
            let mut locked = write_lock!(loss);
            let start_time = Instant::now();
            locked.calc_gradients();
            let finish = Instant::now();
            println!(
                "done: {}s/it ",
                (finish - start_time).as_secs()
            );
            locked.value()
        };

        for param in params_flat.iter() {
            let value = read_lock!(param).value();
            let grad = read_lock!(param).grad().unwrap();
            // println!("param {}(grad={}, alter={})", value, grad, -grad * learning_rate(it));
            write_lock!(param).set_value(value + -grad * learning_rate(it));
        }

        println!("loss={}, lr={}", actual_loss, learning_rate(it));

        visualize(loss);
    }
}
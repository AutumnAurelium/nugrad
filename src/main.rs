use std::{fs::File, io::{BufRead, BufReader}, iter, result::IntoIter, time::Instant};

use nn::*;
use serde::{Deserialize, Serialize};
use serde_json::{de::IoRead, Deserializer};
use value::{as_values, sum, ParentVec, Value, ValueLock};
use viz::visualize;

pub mod value;
pub mod nn;
pub mod viz;

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

    let test_data = load_mnist("data/mnist_handwritten_test.jsonl", 1000);
}

#[cfg(test)]
mod tests {
    use viz::visualize;

    use crate::*;
    use crate::value::*;

    #[test]
    fn test_basic() {
        let eq = add(Value::new(5.0), Value::new(3.0));

        assert_eq!(read_lock!(eq).value(), 8.0);
    }

    #[test]
    fn test_grads() {
        let a = Value::param(5.0);
        let b = Value::param(6.0);

        let eq = mul(a.clone(), b.clone());

        write_lock!(eq).calc_gradients();

        visualize(eq.clone());

        assert_eq!(read_lock!(a).grad().unwrap(), 6.0);
    }

    #[test]
    fn test_neuron() {
        let mut inputs = vec![];
        let mut params = vec![];
        for i in 0..4 {
            inputs.push(Value::new(i as f32));
            params.push(Value::param(i as f32 + 10.0));
        }
        params.push(Value::param(-1.0));

        let neuron = neuron(&inputs, &params, Operator::ReLU);

        write_lock!(neuron).calc_gradients();

        assert_eq!(read_lock!(neuron).value(), 0.0*10.0 + 1.0*11.0 + 2.0*12.0 + 3.0*13.0 - 1.0);
        assert_eq!(read_lock!(params[2]).grad().unwrap(), 2.0);
    }

    #[test]
    fn test_rmse() {
        let output = as_params(&vec![1.0, 3.0]);
        let target = as_values(&vec![1.5, 2.5]);

        let rmse = rmse(output.clone(), target.clone());

        write_lock!(rmse).forward_pass();
        write_lock!(rmse).calc_gradients();

        visualize(rmse.clone());

        assert_eq!(read_lock!(rmse).value(), 0.5);
        assert_eq!(read_lock!(output[0]).grad().unwrap(), -0.5);
        assert_eq!(read_lock!(output[1]).grad().unwrap(), 0.5);
    }

    #[test]
    fn simple_network() {

    }
}
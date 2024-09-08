use rand::{thread_rng, Rng};

use crate::{read_lock, value::*};

/**
 * Creates a single neuron, given a set of inputs and a set of weights.
 */
pub fn neuron(inputs: &ParentVec, params: &ParentVec, activation: Operator) -> ValueLock {
    assert!(activation.input_count_valid(1)); // Activation has to be a unary operator.
    assert_eq!(inputs.len() + 1, params.len()); // Params vec needs to have all the weights plus a bias.

    let mut parents = vec![];
    for i in 0..inputs.len() {
        parents.push(params[i].clone());
        parents.push(inputs[i].clone());
    }
    // Bias
    parents.push(params.last().unwrap().clone());

    return Value::op(activation, vec![
        Value::op(Operator::PartialNeuron, parents)
    ]);
}

/**
 * Creates a single perceptron layer.
 */
pub fn layer(inputs: &ParentVec, layer_params: &Vec<ParentVec>, neuron_count: usize, activation: Operator) -> Vec<ValueLock> {
    assert_eq!(layer_params.len(), neuron_count); // Must have all neuron weights.

    let mut neurons = vec![];

    for i in 0..neuron_count {
        let neuron = neuron(inputs, &layer_params[i], activation);
        neurons.push(neuron);
    }

    return neurons;
}

/**
 * Creates a multi-layer perceptron with a sequence of neuron counts controlled by the `neuron_counts` Vec.
 * Returns a vec of the output values and a Vec<Vec<ParentVec>> of the parameters generated for it.
 */
pub fn mlp(inputs: ParentVec, neuron_counts: Vec<usize>, network_params: &Vec<Vec<ParentVec>>, activation: Operator) -> Vec<ValueLock> {
    assert_eq!(network_params.len(), neuron_counts.len()); // Must have all layer weights.

    let mut prev_inputs = inputs;

    for (i, neuron_count) in neuron_counts.iter().enumerate() {
        let layer = layer(&prev_inputs, &network_params[i], *neuron_count, activation);
        prev_inputs = layer;
    }

    return prev_inputs;
}

/**
 * Generates a set of parameter values for a MLP of a given size.
 */
pub fn mlp_parameters(inputs: usize, neuron_counts: Vec<usize>) -> Vec<Vec<ParentVec>> {
    let mut params = vec![];

    let mut rng = thread_rng();

    let mut last_input_count = inputs;

    for neuron_count in neuron_counts {
        let mut layer_params = vec![];
        for _ in 0..neuron_count {
            let mut neuron_params = vec![];
            for _ in 0..last_input_count {
                neuron_params.push(Value::param(rng.gen()));
            }
            neuron_params.push(Value::param(rng.gen())); // Add bias

            layer_params.push(neuron_params);
        }
        last_input_count = neuron_count;
        params.push(layer_params);
    }

    return params;
}

/**
 * Generates a root-mean-squared error function given a set of outputs and a corresponding set of target values.
 */
pub fn rmse(outputs: Vec<ValueLock>, targets: Vec<ValueLock>) -> ValueLock {
    assert_eq!(outputs.len(), targets.len());

    let mut diffs = vec![];

    for i in 0..outputs.len() {
        diffs.push(pow(sub(targets[i].clone(), outputs[i].clone()), Value::new(2.0)));
    }

    let n = Value::new(outputs.len() as f32);
    return sqrt(
        div(
            sum(diffs),
            n
        )
    );
}
use core::{f32, num};
use std::{borrow::Borrow, cell::RefCell, cmp::max, fmt::Display, ops::{Add, Div, Mul, Sub}, rc::Rc, sync::{Arc, RwLock}, thread};

use rayon::prelude::*;

/**
 * Shortcut for $rwlock.read().unwrap().
 * This is a very low-stakes project; if another thread panics mid-read I just want the program to crash.
 */
#[macro_export]
macro_rules! read_lock {
    ($rwlock: expr) => {
        $rwlock.read().expect("RwLock poisoned!")
    };
}

/**
 * Returns the data value of a given ValueLock and then releases it.
 */
macro_rules! read_data {
    ($rwlock: expr) => {
        {
            $rwlock.read().expect("RwLock poisoned!").value()
        }
    };
}

/**
 * Shortcut for $rwlock.write().unwrap().
 * This is a very low-stakes project; if another thread panics mid-read I just want the program to crash.
 */
#[macro_export]
macro_rules! write_lock {
    ($rwlock: expr) => {
        $rwlock.write().expect("RwLock poisoned!")
    };
}

pub type ValueLock = Arc<RwLock<Value>>;
pub type ParentVec = Vec<ValueLock>;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Operator {
    Variable,
    Add, Subtract, Multiply, Divide, Power, // Simple binary ops
    Negate, ReLU, TanH, Sqrt, // Unary ops
    Sum,
    PartialNeuron
}

impl Operator {
    pub fn input_count_valid(&self, num: usize) -> bool {
        match self {
            Operator::Variable => num == 0,
            Operator::Add => num == 2,
            Operator::Subtract => num == 2,
            Operator::Multiply => num == 2,
            Operator::Divide => num == 2,
            Operator::Power => num == 2,
            Operator::Negate => num == 1,
            Operator::ReLU => num == 1,
            Operator::TanH => num == 1,
            Operator::Sqrt => num == 1,
            Operator::Sum => num != 0,
            Operator::PartialNeuron => num >= 3 && num%2 == 1, // a neuron must have at least 1 input, 1 weight, and 1 bias. this will always be non-even.
        }
    }

    fn forward(&self, parents: &ParentVec) -> f32 {
        assert!(self.input_count_valid(parents.len()));
        match self {
            Operator::Variable => panic!("Can't do forward() on a Variable."),
            Operator::Add => read_data!(parents[0]) + read_data!(parents[1]),
            Operator::Subtract => read_data!(parents[0]) - read_data!(parents[1]),
            Operator::Multiply => read_data!(parents[0]) * read_data!(parents[1]),
            Operator::Divide => read_data!(parents[0]) / read_data!(parents[1]),
            Operator::Power => read_data!(parents[0]).powf(read_data!(parents[1])),
            Operator::Negate => -read_data!(parents[0]),
            Operator::ReLU => read_data!(parents[0]).max(0.0),
            Operator::TanH => read_data!(parents[0]).tanh(),
            Operator::Sqrt => read_data!(parents[0]).sqrt(),
            Operator::Sum => {
                let mut sum = 0.0;
                for value in parents {
                    sum += read_lock!(value).data;
                }
                sum
            },
            Operator::PartialNeuron => {
                let mut weighted_sum = 0.0;
                for chunk in parents.chunks_exact(2) {
                    let weight = &chunk[0];
                    let input = &chunk[1];
                    weighted_sum += read_data!(weight) * read_data!(input);
                }

                weighted_sum + read_data!(parents.last().unwrap())
            },
        }
    }

    fn back(&self, value: &Value, input: usize) -> f32 {
        // I'm going to assume all of these values have the right parent Vec, because otherwise how did you do the forward pass?
        value.grad * match self {
            Operator::Variable => 1.0,
            Operator::Add => 1.0,
            Operator::Subtract => {
                if input == 0 {
                    1.0
                } else {
                    -1.0
                }
            },
            Operator::Multiply => {
                if input == 0 {
                    read_data!(value.parents[1])
                } else {
                    read_data!(value.parents[0])
                }
            },
            Operator::Divide => {
                let a = read_data!(value.parents[0]);
                let b = read_data!(value.parents[1]);

                if input == 0 {
                    1.0 / b
                } else {
                    (-a) / (b.powi(2))
                }
            },
            Operator::Power => {
                let a = read_data!(value.parents[0]);
                let b = read_data!(value.parents[1]);

                if input == 0 {
                    b * a.powf(b - 1.0)
                } else {
                    // This is NaN if A is negative!
                    a.powf(b) * a.ln()
                }
            },
            Operator::Negate => -1.0,
            Operator::ReLU => {
                // We arbitrarily assign the derivative of ReLU at 0 to be 0.
                if read_data!(value.parents[0]) > 0.0 {
                    1.0
                } else {
                    0.0
                }
            },
            Operator::TanH => 1.0 - read_data!(value.parents[0]).tanh().powi(2),
            Operator::Sqrt => 1.0 / (2.0 * read_data!(value.parents[0]).sqrt()),
            Operator::Sum => 1.0,
            Operator::PartialNeuron => {
                // If this is the bias
                if input == value.parents.len()-1 {
                    1.0
                } else {
                    if input % 2 == 0 {
                        read_data!(value.parents[input+1])
                    } else {
                        read_data!(value.parents[input-1])
                    }
                }
            },
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct Value {
    data: f32,
    op: Operator,
    parents: ParentVec,

    grad: f32,
    needs_grad: bool,
}

impl Value {
    /**
     * Creates a variable that does not need to have gradients calculated.
     */
    pub fn new(data: f32) -> ValueLock {
        Arc::new(RwLock::new(Value {
            data,
            op: Operator::Variable,
            parents: vec![],
            grad: f32::NAN,
            needs_grad: false
        }))
    }

    /**
     * Creates a parameter, which will have gradients calculated.
     */
    pub fn param(data: f32) -> ValueLock {
        Arc::new(RwLock::new(Value {
            data,
            op: Operator::Variable,
            parents: vec![],
            grad: f32::NAN,
            needs_grad: true
        }))
    }

    /**
     * Creates a value with a given parameter and set of inputs.
     */
    pub fn op(op: Operator, parents: ParentVec) -> ValueLock {
        let mut needs_grad = false;
        for parent in parents.iter() {
            if parent.read().unwrap().needs_grad {
                needs_grad = true;
                break;
            }
        }

        Arc::new(RwLock::new(Value {
            data: op.forward(&parents),
            op,
            parents,
            grad: f32::NAN,
            needs_grad
        }))
    }

    /**
     * Returns the cached data of this value. May be incorrect if the expression graph is dirty.
     */
    pub fn value(&self) -> f32 {
        self.data
    }

    /**
     * Sets the value if this is a Variable. Otherwise, panics.
     */
    pub fn set_value(&mut self, value: f32) {
        if matches!(self.op, Operator::Variable) {
            self.data = value;
        } else {
            panic!("Tried to set value on a non-variable.");
        }
    }

    /**
     * Returns Some(grad) if the gradient is set, otherwise None.
     */
    pub fn grad(&self) -> Option<f32> {
        if self.grad.is_nan() {
            return None;
        } else {
            return Some(self.grad);
        }
    }

    /**
     * Returns the value of needs_grad
     */
    pub fn needs_grad(&self) -> bool {
        return self.needs_grad;
    }

    /**
     * Returns a reference to the ParentVec representing the parent values.
     */
    pub fn parents(&self) -> &ParentVec {
        return &self.parents;
    }

    /**
     * Returns the operator used by this value.
     */
    pub fn operator(&self) -> Operator {
        return self.op;
    }

    /**
     * Returns a Vec of references to every Variable value below this one in the expresssion graph.
     */
    pub fn all_inputs(&self) -> ParentVec {
        let mut inputs = vec![];

        for parent in self.parents.iter() {
            if matches!(read_lock!(parent).op, Operator::Variable) {
                inputs.push(parent.clone());
            } else {
                for input in read_lock!(parent).all_inputs() {
                    inputs.push(input);
                }
            }
        }

        return inputs;
    }

    /**
     * Returns a Vec of mutable references to every Variable value below this one in the expresssion graph.
     */
    pub fn all_inputs_mut(&mut self) -> ParentVec {
        let mut inputs = vec![];

        for parent in self.parents.iter() {
            if matches!(read_lock!(parent).op, Operator::Variable) {
                inputs.push(parent.clone());
            } else {
                for input in write_lock!(parent).all_inputs_mut() {
                    inputs.push(input);
                }
            }
        }

        return inputs;
    }

    /**
     * Performs the forward pass.
     */
    pub fn forward_pass(&mut self) {
        self.invalidate_recursive();
        self.forward_pass_recursive();
    }

    /**
     * Recursively updates every value below this value in the expression graph.
     */
    pub fn forward_pass_recursive(&mut self) {
        if self.data.is_nan() {
            maybe_parallel(&self.parents, self.parents.len() > 2, |parent| {
                write_lock!(parent).forward_pass_recursive();
            });

            self.data = self.op.forward(&self.parents);
        }
    }

    /**
     * Recursively invalidates the values of every value below this one in the expression graph.
     */
    fn invalidate_recursive(&mut self) {
        if !matches!(self.op, Operator::Variable) {
            if self.data != f32::NAN {
                self.data = f32::NAN;
                maybe_parallel(&self.parents, self.parents.len() > 2, |parent| {
                    write_lock!(parent).invalidate_recursive();
                });
            }
        }
    }

    /**
     * Recursively calculates the gradient of each value below this one in the expression graph, relative to this value.
     */
    pub fn calc_gradients(&mut self) {
        // self.clear_grads_recursive();

        self.grad = 1.0; // By definition, our gradient with respect to ourself is 1.

        self.calc_gradients_recursive();
    }

    /**
     * Resets the gradients of all values below this one in the expression graph.
     */
    fn clear_grads_recursive(&mut self) {
        if self.needs_grad {
            self.grad = 0.0;
        } else {
            self.grad = f32::NAN;
        }
        
        for parent_lock in self.parents.iter_mut() {
            write_lock!(parent_lock).clear_grads_recursive();
        }
    }

    /**
     * Recursively calculates the gradient of each value below this one in the expression graph, using this value's gradient to perform the chain rule.
     */
    fn calc_gradients_recursive(&mut self) {
        for (i, parent) in self.parents.iter().enumerate() {
            if !read_lock!(parent).needs_grad {
                continue;
            }

            // We add here to account for the effects of values being used more than once.
            let new_grad = self.op.back(self, i);

            // let mut lock = write_lock!(parent);

            if read_lock!(parent).grad.is_nan() {
                write_lock!(parent).grad = new_grad;
            } else {
                write_lock!(parent).grad += new_grad;
            }

            write_lock!(parent).calc_gradients_recursive();
        };
    }

    /**
     * Returns the depth of the expression graph starting at this node.
     */
    pub fn depth(&self) -> usize {
        self.parents.iter().map(|p| read_lock!(p).depth()+1).max().unwrap_or(0)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.op {
            Operator::Variable => write!(f, "[{}, {}]", self.data, self.grad),
            other => {
                write!(f, "{:?}=[{}, {}]", other, self.data, self.grad)
            }
        }
    }
}

// Various operator convenience functions
pub fn add(lhs: ValueLock, rhs: ValueLock) -> ValueLock {
    Value::op(Operator::Add, vec![lhs, rhs])
}

pub fn sub(lhs: ValueLock, rhs: ValueLock) -> ValueLock {
    Value::op(Operator::Subtract, vec![lhs, rhs])
}

pub fn mul(lhs: ValueLock, rhs: ValueLock) -> ValueLock {
    Value::op(Operator::Multiply, vec![lhs, rhs])
}

pub fn div(numerator: ValueLock, denominator: ValueLock) -> ValueLock {
    Value::op(Operator::Divide, vec![numerator, denominator])
}

pub fn pow(base: ValueLock, exponent: ValueLock) -> ValueLock {
    Value::op(Operator::Power, vec![base, exponent])
}

pub fn sqrt(value: ValueLock) -> ValueLock {
    Value::op(Operator::Sqrt, vec![value])
}

pub fn sum(values: ParentVec) -> ValueLock {
    return Value::op(
        Operator::Sum,
        values
    )
}

pub fn as_values(values: &Vec<f32>) -> ParentVec {
    return values.iter().map(|v| Value::new(*v)).collect();
}

pub fn as_params(values: &Vec<f32>) -> ParentVec {
    return values.iter().map(|v| Value::param(*v)).collect();
}

fn maybe_parallel(data: &ParentVec, condition: bool, each: fn(&ValueLock)) {
    if condition {
        data.par_iter().for_each(|x| each(x));
    } else {
        data.iter().for_each(|x| each(x));
    }
}
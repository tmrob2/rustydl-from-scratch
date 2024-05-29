#![allow(non_snake_case)]
use crate::metrics::r2_score;
use num_traits::ToPrimitive;
use numpy::{ndarray::{Array2, ArrayView2, Axis}, IntoPyArray, PyArray2, PyReadonlyArray2};
use rand_distr::StandardNormal;
use rand::prelude::*;
use pyo3::prelude::*;

struct ForwardPass<'a> {
    pub X: ArrayView2<'a, f32>,
    pub y: ArrayView2<'a, f32>,
    pub M1: Array2<f32>,
    pub N1: Array2<f32>,
    pub O1: Array2<f32>,
    pub M2: Array2<f32>,
    pub P: Array2<f32>,
    pub loss: f32
}

struct Weights {
    W1: Array2<f32>,
    B1: Array2<f32>,
    W2: Array2<f32>,
    B2: Array2<f32>
}

fn initialise_weights(input_size: usize, hidden_size: usize) -> Weights {
    //let weights: Vec<f32> = ...;
    let sample_weights1: Vec<f32> = (0..input_size * hidden_size)
        .map(|_| thread_rng().sample(StandardNormal)).collect();
    let sample_weights2: Vec<f32> = (0..hidden_size)
        .map(|_| thread_rng().sample(StandardNormal)).collect();
    let sample_weights_b1: Vec<f32> = (0..hidden_size)
        .map(|_| thread_rng().sample(StandardNormal)).collect();
    let W1: Array2<f32> = Array2::from_shape_vec((input_size, hidden_size), sample_weights1)
        .unwrap();
    let W2: Array2<f32> = Array2::from_shape_vec((hidden_size,1), sample_weights2)
        .unwrap();
    let B1: Array2<f32> = Array2::from_shape_vec((1, hidden_size),sample_weights_b1)
        .unwrap();
    let B2: Array2<f32> = Array2::from_shape_vec((1, 1), vec![thread_rng().sample(StandardNormal)])
        .unwrap();
    Weights {
        W1,
        B1,
        W2, 
        B2
    }
}

fn forward_loss<'a, F>(X: ArrayView2<'a, f32>, y: ArrayView2<'a, f32>, weights: &Weights, activation_fn: F) 
-> ForwardPass<'a>
where F: Fn(f32) -> f32 {
    let M1: Array2<f32>   = X.dot(&weights.W1);
    let N1: Array2<f32>   = &M1 + &weights.B1;
    let O1: Array2<f32>   = N1.map(|x| activation_fn(*x));
    let M2: Array2<f32>   = O1.dot(&weights.W2);
    let P: Array2<f32>    = &M2 + &weights.B2;
    // Get the mean squared error
    let loss: f32 = (&y - &P).map(|x| x * x).sum() / y.nrows().to_f32().unwrap();

    ForwardPass {
        X,
        y,
        M1,
        N1,
        O1,
        M2,
        P,
        loss
    }
}

fn loss_gradients<F>(forward_info: &ForwardPass, weights: &Weights, activation_fn: F) -> Weights
where F: Fn(f32) -> f32 {
    // A function to compute the partial derivatives of the loss with respect to 
    // each of the parameters in the neural network

    let dLdP: Array2<f32> = -(&forward_info.y - &forward_info.P);
    let dPdM2: Array2<f32> = Array2::from_elem(
        (forward_info.M2.nrows(), forward_info.M2.ncols()), 1f32);
    let dLdM2: Array2<f32> = &dLdP * &dPdM2;
    let dPdB2: Array2<f32> = Array2::from_elem(
        (weights.B2.nrows(), weights.B2.ncols()), 1f32);
    let dLdB2: Array2<f32> = (&dLdP * &dPdB2).sum_axis(Axis(0)).insert_axis(Axis(1));

    let dM2dW2: ArrayView2<f32> = forward_info.O1.t();
    let dLdW2: Array2<f32> = dM2dW2.dot(&dLdP);

    let dM2dO1: ArrayView2<f32> = weights.W2.t();
    let dLdO1: Array2<f32> = dLdM2.dot(&dM2dO1);

    let dO1dN1: Array2<f32> = forward_info.N1.map(|x| activation_fn(*x));
    let dLdN1 = dLdO1 * &dO1dN1;

    let dN1dB1: Array2<f32> = Array2::from_elem(
        (weights.B1.nrows(), weights.B1.ncols()), 1f32);
    
    let dN1dM1: Array2<f32> = Array2::from_elem(
        (forward_info.M1.nrows(), forward_info.M1.ncols()), 1f32);
    
    let dLdB1 = (&dLdN1 * &dN1dB1).sum_axis(Axis(0)).insert_axis(Axis(0));

    let dLdM1: Array2<f32> = &dLdN1 * &dN1dM1;

    let dM1dW1: ArrayView2<f32> = forward_info.X.t();

    let dLdW1 = dM1dW1.dot(&dLdM1);

    Weights {
        W2: dLdW2,
        B2: dLdB2,
        W1: dLdW1,
        B1: dLdB1
    }
}

#[pyclass]
struct MLP {
    losses:Vec<f32>,
    val_score:Vec<f32>,
    batch_permutation: Vec<usize>,
    activation_fn: Option<fn(f32) -> f32>,
    weights: Option<Weights>
}

#[pymethods]
impl MLP {
    #[new]
    fn new() -> Self {
        MLP {
            losses: Vec::new(),
            val_score: Vec::new(),
            batch_permutation: Vec::new(),
            activation_fn: None,
            weights: None
        }
    }

    pub fn fit(&mut self, X: PyReadonlyArray2<f32>, y: PyReadonlyArray2<f32>, Xtest: PyReadonlyArray2<f32>,
        ytest: PyReadonlyArray2<f32>, activation_fn: &str, n_iter: usize, test_every: usize, hidden_size: usize,
        learning_rate: f32, batch_size: usize, return_losses: bool, seed: u64) {
        self.activation_fn = match activation_fn {
            "sigmoid" => Some(crate::sigmoidV2),
            _ => None
        };
        self._fit(X.as_array().view(), y.as_array().view(), Xtest.as_array().view(), ytest.as_array().view(), 
                  n_iter, test_every, hidden_size, learning_rate, batch_size, return_losses, seed);
    } 

    pub fn predict<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f32>) -> Bound<'py, PyArray2<f32>> {

        let pred = self._predict(X.as_array().view(), &self.activation_fn.unwrap());
        pred.into_pyarray_bound(py)
    }
}

impl MLP {
    pub fn permute_data(&mut self, nrows: usize, seed: u64) {
        let mut new_order: Vec<usize> = (0..nrows).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        new_order.as_mut_slice().shuffle(&mut rng);
        self.batch_permutation = new_order;
    }

    fn _fit(&mut self, X: ArrayView2<f32>, y: ArrayView2<f32>, Xtest: ArrayView2<f32>, ytest: ArrayView2<f32>, 
        n_iter: usize, test_every: usize, hidden_size: usize, learning_rate:f32, 
        mut batch_size: usize, return_losses: bool, seed: u64){
        /*
        Train the model for a number of epochs - this is obviously redundant because the NN
        architecture is static, i.e. we cannot generate a general computational graph. This 
        will be the precursor to a neural network framework which will be able to handle 
        any computational graph.
         */

        let mut start: usize = 0;
        
        // initialise the weights
        let mut weights = initialise_weights(X.ncols(), hidden_size);

        self.permute_data(X.nrows(), seed); // assigns an initalised random order to self

        for i in 0..n_iter {
            
            if start >= X.nrows() {
                self.permute_data(X.nrows(), seed);
                start = 0;
            }
            // generate a batch 
            if start + batch_size > X.nrows() {
                // basically we select slices of the data according to the batch
                // size and then iterate trhough
                batch_size = X.nrows() - start;
            }

            // select a segment of the random permutation of indices
            let batch_selection = &self.batch_permutation[start..(start + batch_size)];

            // slide the window forward
            start += batch_size;

            // Train based on the generated batch
            let X_: Array2<f32> = X.select(Axis(0), &batch_selection);
            let y_: Array2<f32> = y.select(Axis(0), &batch_selection);
            let forward_info = forward_loss(X_.view(), y_.view(), &weights, 
                &self.activation_fn.unwrap());

            let loss_gradients = loss_gradients(&forward_info, &weights, 
                &self.activation_fn.unwrap());

            if return_losses {
                self.losses.push(forward_info.loss);
            }

            // Update each member of the struct
            weights.W1 -= &(learning_rate * loss_gradients.W1);
            weights.B1 -= &(learning_rate * loss_gradients.B1);
            weights.W2 -= &(learning_rate * loss_gradients.W2);
            weights.B2 -= &(learning_rate * loss_gradients.B2);

            if i % test_every == 0 {
                let preds: Array2<f32> = self._predict(Xtest, &self.activation_fn.unwrap());
                self.val_score.push(r2_score(preds.view(), ytest))
            }
        }  
        
    }

    fn _predict<F>(&mut self, X: ArrayView2<f32>, activation_fn: F) 
    -> Array2<f32>
    where F: Fn(f32)->f32 {
        // Generate predictions from the step to step neural network model
        let M1: Array2<f32> = X.dot(&self.weights.as_ref().unwrap().W1);

        let N1: Array2<f32> = M1 + &self.weights.as_ref().unwrap().B1; 
        // The above two allocations can actually just be turned into reference operations
        // i.e N1 = &X.dot(&weights.W1) + &weights.B1 but for readibility it makes sense to 
        // keep this as it is
        let O1: Array2<f32> = N1.map(|x| activation_fn(*x));
        let M2: Array2<f32> = O1.dot(&self.weights.as_ref().unwrap().W2);
        M2 + &self.weights.as_ref().unwrap().B2 // -> P
    }
}


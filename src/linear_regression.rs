#[allow(non_snake_case)]
/*
The training function for linear regression has the following workflow:

0. Permute the data - random shuffle
1. batch generation
2. forward loss: forward pass
3. loss_gradients: backward pass
4. update the loss_grads with a learning rate

*/

//use nalgebra::{DMatrix, DMatrixView};
use numpy::ndarray::{Array2, ArrayView2, Axis};
use numpy::PyReadonlyArrayDyn;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{pyclass, pymethods, Bound, Python};
use rand_distr::StandardNormal;
use rand::prelude::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;


#[pyclass]
pub struct LinearRegression {
    pub losses: Vec<f32>,
    weights: Weights,
    batch_permutation: Vec<usize>
}

#[pymethods]
impl LinearRegression {
    #[new]
    pub fn new() -> Self {
        LinearRegression {
            losses: Vec::new(),
            weights: Weights::new(),
            batch_permutation: Vec::new()
        }
    }

    pub fn fit(&mut self, X: PyReadonlyArray2<f32>, y: PyReadonlyArray2<f32>, 
               n_iter: usize, batch_size: usize, learning_rate: f32, return_losses:bool,
               seed: u64) {
        self._fit(X.as_array().view(), y.as_array().view(), 
                  n_iter, learning_rate, batch_size, return_losses, seed);
    }

    pub fn predict<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f32>) -> Bound<'py, PyArray2<f32>> {
        let pred = self._predict(X.as_array().view());
        pred.into_pyarray_bound(py)
    }

}

impl LinearRegression {

    pub fn permute_data(&mut self, nrows: usize, seed: u64) {
        let mut new_order: Vec<usize> = (0..nrows).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        new_order.as_mut_slice().shuffle(&mut rng);
        self.batch_permutation = new_order;
    }

    fn _fit(&mut self, X: ArrayView2<f32>, y: ArrayView2<f32>, n_iter: usize, 
        learning_rate: f32, mut batch_size: usize, return_losses: bool, seed: u64) {
        /*
        Train the model for a number of epochs
        */
        let mut start: usize = 0;

        // Initialise the weights
        let mut weights = initialise_weights(X.ncols());
        
        let mut losses: Vec<f32> = if return_losses {
            Vec::with_capacity(n_iter)
        } else {
            Vec::new()
        };

        // generate a random permutation of indices based on the size of X
        self.permute_data(X.nrows(), seed);

        for _ in 0..n_iter {
            // if the available index selection is smaller than the batch size, then
            // reselect the random ordering
            if start >= X.nrows() {
                self.permute_data(X.nrows(), seed);
                start = 0; // reset start here
            } 
            
            // Generate a batch
            if start + batch_size > X.nrows() {
                // should really throw an error here because this means that the batch
                // size is too large...
                batch_size = X.nrows() - start;
            }

            // select a segment of the random permuation of indices
            let batch_selection = &self.batch_permutation[start..(start + batch_size)];
            
            // slide the window forward
            start += batch_size;
            
            // Train based on the generated batch
            // TODO Personally, I don't like this implementation but I can't think of anything else at the moment
            //  The problem is that the data matrix is being copied each time that this batch selection is going on
            //  This does not seem like an efficient way of solving this problem
            let X_: Array2<f32> = X.select(Axis(0), &batch_selection);
            let y_: Array2<f32> = y.select(Axis(0), &batch_selection); 
            let forward_info = forward_loss(X_.view(), y_.view(), &weights);
            
            if return_losses {
                losses.push(forward_info.loss);
            }
            
            let loss_weights = loss_gradients(&forward_info, &weights);
            //println!("Return from loss => W: {:?}, B: {:?}", loss_weights.W.shape(), loss_weights.B.shape());
            //println!("W:\n{}\nB:\n{}", loss_weights.W, loss_weights.B);
            weights.W -= &(learning_rate * loss_weights.W);
            weights.B -= &(learning_rate * loss_weights.B);
        }

        self.losses = losses;
        self.weights = weights;
        
    }

    fn _predict(&mut self, X: ArrayView2<f32>) -> Array2<f32> {
        let N: Array2<f32> = X.dot(&self.weights.W);
        N + &self.weights.B
    }
}


pub struct ForwardPass<'a> {
    pub X: ArrayView2<'a, f32>,
    pub y: ArrayView2<'a, f32>,
    pub P: Array2<f32>,
    pub N: Array2<f32>,
    pub loss: f32
}

struct Weights {
    W: Array2<f32>,
    B: Array2<f32>
}

impl Weights {
    pub fn new() -> Self {
        Weights {
            W: Array2::default((0, 0)),
            B: Array2::default((0, 0))
        }
    }
}

// A function which computes the loss of the forward pass through the computational graph for
// linear regression. Where possible we use matrix views so that we are not creating memory
// allocations. 
fn forward_loss<'a>(X: ArrayView2<'a, f32>, 
                        y: ArrayView2<'a, f32>, 
                        weights: &Weights) -> ForwardPass<'a> {
    // Make sure that all of the dimensions align correctly
    //println!("Shape X: {:?}, shape y: {:?}, shape W: {:?}, B: {:?}", 
    //    X.shape(), y.shape(), weights.W.shape(), weights.B.shape());

    assert_eq!(X.nrows(), y.nrows(), "The rows of X do not match the rows of y");
    assert_eq!(X.ncols(), weights.W.nrows(), "The cols of X does not match the rows of W");
    assert_eq!(weights.B.nrows(), weights.B.ncols(), "B is not a scalar with a single dimenstion");
    //println!("attempting to multiply X with W");
    let N: Array2<f32> = X.dot(&weights.W);
    //println!("attempting to add X.W to B");
    let P: Array2<f32> = &N + &weights.B;
    //println!("attempting to sum the rows of y with P, P shape: {:?}", P.shape());
    let loss: f32 = (&y + &P).map(|x| x*x).sum() / (y.nrows() as f32);
    ForwardPass {
        X,
        y,
        P,
        N,
        loss
    }
}

// Initialise the weights of the first forward pass of the model
fn initialise_weights(n_in: usize) -> Weights {

    let weights: Vec<f32> = (0..n_in).map(|_| thread_rng().sample(StandardNormal)).collect();
    let W: Array2<f32> = Array2::from_shape_vec((n_in, 1), weights).unwrap();
    let B: Array2<f32> = Array2::from_shape_vec((1,1),vec![thread_rng().sample(StandardNormal)]).unwrap();
    Weights {
        W,
        B
    }
}

fn loss_gradients(forward_info: &ForwardPass, weights: &Weights) -> Weights {
    //let batch_size = forward_info.X.nrows();

    // L = Lambda(P, Y) = (Y - P)^2 -> dL/dP = -2 * (Y - P)
    //println!("Attempting to substract p from y in forward info");
    let dLdP: Array2<f32> = -2.0 * (&forward_info.y - &forward_info.P);
    // P = N + B -> dP/dN = 1s 
    let dPdN: Array2<f32> = Array2::from_elem((forward_info.N.nrows(),forward_info.N.ncols()),1f32);
    // P = N + B -> dP/dB = 1s
    let dPdB: Array2<f32> = Array2::from_elem((weights.B.nrows(), weights.B.ncols()), 1f32);
    // chain rule
    //println!("dLdP shape: {:?}, dPdN shape: {:?}", dLdP.shape(), dPdN.shape());
    let dLdN: Array2<f32> = &dLdP * &dPdN;

    // Now N = X . W -> dNdW = X^T
    let dNdW: ArrayView2<f32> = forward_info.X.t();
    
    //println!("dNdW shape: {:?}, dLdN shape: {:?}", dNdW.shape(), dLdN.shape());
    // chain rule all the way back
    let dLdW: Array2<f32> = dNdW.dot(&dLdN);
    
    //println!("dLdP shape: {:?}, dPdB shape: {:?}", &dLdP.shape(), dPdB.shape());
    // chain rule to B
    let row_vector = (dLdP * &dPdB).sum_axis(Axis(0));

    // currently representing this a column vector
    Weights {
        W: dLdW,
        B: row_vector.insert_axis(Axis(0)) // unfortunately this is a Array1 so we need to do a conversion to array2
    }
}


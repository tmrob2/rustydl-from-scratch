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
use ndarray::{Array2, ArrayView2, s, Axis};
use rand_distr::StandardNormal;
use rand::prelude::*;

pub struct Batch<'a> {
    pub X: ArrayView2<'a, f32>,
    pub y: ArrayView2<'a, f32>
}

pub struct ForwardPass<'a> {
    pub X: ArrayView2<'a, f32>,
    pub y: ArrayView2<'a, f32>,
    pub P: Array2<f32>,
    pub N: Array2<f32>,
    pub loss: f32
}

pub struct Weights {
    pub W: Array2<f32>,
    pub B: Array2<f32>
}

// A function which generates a batch from a matrix which is stored on the heap. The
// function is efficient in the sense that is returns a Batch<'a> which is just a 
// view of the matrices X, and y
// X: 2-dimensional matrix of type f32 -> TODO could generalise the number type
// y: 2-dimensional matrix -> Really this is a column vector (n, 1) dimensional matrix, always. 
// Returns a Batch<'a> for whos elements lives the same lifetime as the matrices X, y
// A batch consists of a X, y -> feature matrix, target matrix resp. 
pub fn generate_batch<'a>(X: &'a Array2<f32>, // TODO possibly change the type here to view
                          y: &'a Array2<f32>, // ^
                          start: usize, 
                          mut batch_size: usize) -> Batch<'a> {
    if start + batch_size > X.nrows() {
        batch_size = X.nrows() - start;
    }
    Batch {
        X: X.slice(s![start..batch_size,..]),
        y: y.slice(s![start..batch_size,..])
    }
}

// A function which computes the loss of the forward pass through the computational graph for
// linear regression. Where possible we use matrix views so that we are not creating memory
// allocations. 
pub fn forward_loss<'a>(X: ArrayView2<'a, f32>, 
                        y: ArrayView2<'a, f32>, 
                        weights: &Weights) -> ForwardPass<'a> {
    // Make sure that all of the dimensions align correctly
    assert_eq!(X.ncols(), y.nrows());
    assert_eq!(X.ncols(), weights.W.nrows());
    assert_eq!(weights.B.nrows(), weights.B.ncols());

    let N: Array2<f32> = X.dot(&weights.W);
    let P: Array2<f32> = &N + &weights.B;
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
pub fn initialise_weights(n_in: usize) -> Weights {

    let weights: Vec<f32> = (0..n_in).map(|_| thread_rng().sample(StandardNormal)).collect();
    let W: Array2<f32> = Array2::from_shape_vec((n_in, 1), weights).unwrap();
    let B: Array2<f32> = Array2::from_shape_vec((1,1),vec![thread_rng().sample(StandardNormal)]).unwrap();
    Weights {
        W,
        B
    }
}

pub fn loss_gradients(forward_info: &ForwardPass, weights: &Weights) -> Weights {
    //let batch_size = forward_info.X.nrows();

    // L = Lambda(P, Y) = (Y - P)^2 -> dL/dP = -2 * (Y - P)
    let dLdP: Array2<f32> = -2.0 * (&forward_info.y - &forward_info.P);
    // P = N + B -> dP/dN = 1s 
    let dPdN: Array2<f32> = Array2::from_elem((forward_info.N.nrows(),forward_info.N.ncols()),1f32);
    // P = N + B -> dP/dB = 1s
    let dPdB: Array2<f32> = Array2::from_elem((weights.B.nrows(), weights.B.ncols()), 1f32);
    // chain rule
    let dLdN: Array2<f32> = &dLdP * dPdN;

    // Now N = X . W -> dNdW = X^T
    let dNdW: ArrayView2<f32> = forward_info.X.t();

    // chain rule all the way back
    let dLdW: Array2<f32> = &dNdW * &dLdN;
    
    // chain rule to B
    let row_vector = (dLdP * dPdB).sum_axis(Axis(1));
    // currently representing this a column vector
    Weights {
        W: dLdW,
        B: row_vector.into_shape((1, 1)).unwrap() // unfortunately this is a Array1 so we need to do a conversion to array2
    }
}


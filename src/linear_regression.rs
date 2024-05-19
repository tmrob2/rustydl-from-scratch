#[allow(non_snake_case)]
/*
The training function for linear regression has the following workflow:

0. Permute the data - random shuffle
1. batch generation
2. forward loss: forward pass
3. loss_gradients: backward pass
4. update the loss_grads with a learning rate

*/

use nalgebra::{DMatrix, DMatrixView};
use rand_distr::StandardNormal;
use rand::prelude::*;

pub struct Batch<'a> {
    pub X: DMatrixView<'a, f32>,
    pub y: DMatrixView<'a, f32>
}

pub struct ForwardPass<'a> {
    pub X: DMatrixView<'a, f32>,
    pub y: DMatrixView<'a, f32>,
    pub P: DMatrix<f32>,
    pub N: DMatrix<f32>,
    pub loss: f32
}

pub struct Weights {
    pub W: DMatrix<f32>,
    pub B: DMatrix<f32>
}

// A function which generates a batch from a matrix which is stored on the heap. The
// function is efficient in the sense that is returns a Batch<'a> which is just a 
// view of the matrices X, and y
// X: 2-dimensional matrix of type f32 -> TODO could generalise the number type
// y: 2-dimensional matrix -> Really this is a column vector (n, 1) dimensional matrix, always. 
// Returns a Batch<'a> for whos elements lives the same lifetime as the matrices X, y
// A batch consists of a X, y -> feature matrix, target matrix resp. 
pub fn generate_batch<'a>(X: &'a DMatrix<f32>, 
                          y: &'a DMatrix<f32>, 
                          start: usize, 
                          mut batch_size: usize) -> Batch<'a> {
    if start + batch_size > X.nrows() {
        batch_size = X.nrows() - start;
    }
    Batch {
        X: X.view((start, 0), (batch_size, X.ncols())),
        y: y.view((start, 0), (batch_size, 0))
    }
}

// A function which computes the loss of the forward pass through the computational graph for
// linear regression. Where possible we use matrix views so that we are not creating memory
// allocations. 
pub fn forward_loss<'a>(X: DMatrixView<'a, f32>, 
                        y: DMatrixView<'a, f32>, 
                        weights: &Weights) -> ForwardPass<'a> {
    // Make sure that all of the dimensions align correctly
    assert_eq!(X.ncols(), y.nrows());
    assert_eq!(X.ncols(), weights.W.nrows());
    assert_eq!(weights.B.nrows(), weights.B.ncols());

    let N: DMatrix<f32> = X * &weights.W;
    let P: DMatrix<f32> = &N + &weights.B;
    let loss: f32 = (y - &P).map(|x| x*x).sum() / (y.nrows() as f32);
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
    let W: DMatrix<f32> = DMatrix::from_row_slice(n_in, 1, &weights);
    let B: DMatrix<f32> = DMatrix::from_row_slice(
        1, 
        1, 
        &vec![thread_rng().sample(StandardNormal)]
    );
    Weights {
        W,
        B
    }
}

pub fn loss_gradients(forward_info: &ForwardPass, weights: &Weights) -> Weights {
    //let batch_size = forward_info.X.nrows();

    // L = Lambda(P, Y) = (Y - P)^2 -> dL/dP = -2 * (Y - P)
    let dLdP: DMatrix<f32> = -2.0 * (forward_info.y - &forward_info.P);
    // P = N + B -> dP/dN = 1s 
    let dPdN: DMatrix<f32> = DMatrix::from_element(forward_info.N.nrows(), 
                                                   forward_info.N.ncols(),
                                                    1f32);
    // P = N + B -> dP/dB = 1s
    let dPdB: DMatrix<f32> = DMatrix::from_element(weights.B.nrows(), 
                                                   weights.B.ncols(), 
                                                   1f32);
    // chain rule
    let dLdN: DMatrix<f32> = &dLdP * dPdN;

    // Now N = X . W -> dNdW = X^T
    let dNdW: DMatrix<f32> = forward_info.X.transpose();

    // chain rule all the way back
    let dLdW: DMatrix<f32> = dNdW * dLdN;
    
    // chain rule to B
    let row_vector = (dLdP * dPdB).row_sum();
    // currently representing this a column vector
    let dLdB: DMatrix<f32> = DMatrix::from_row_slice(row_vector.len(),
                                                    1, 
                                                    &row_vector.as_slice());
    Weights {
        W: dLdW,
        B: dLdB
    }
}


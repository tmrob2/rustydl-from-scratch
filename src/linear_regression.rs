/*
The training function for linear regression has the following workflow:

0. Permute the data - random shuffle
1. batch generation
2. forward loss: forward pass
3. loss_gradients: backward pass
4. update the loss_grads with a learning rate

*/

use std::collections::HashMap;

use nalgebra::DMatrix;

struct Batch<'a> {
    X: &'a DMatrix<f32>,
    y: &'a DMatrix<f32>
}

// we could even have an implementation here for the Batch but for now just see what we can do with this

/*pub fn generate_batch<'a>(X: &'a DMatrix<f32>, 
                          y: &'a DMatrix<f32>, 
                          start: usize, 
                          batch_size: usize) -> Batch<'a> {
    if start + batch_size > X.nrows() {
        batch_size = X.nrows() - start;
    }
    // Matrix slice
    X.slic
    Batch
}*/


pub fn forward_loss<'a>(X: &'a DMatrix<f32>, 
                        y: &'a DMatrix<f32>, 
                        weights: &HashMap<String, &'a DMatrix<f32>>) {
    assert_eq!(X.ncols(), y.nrows());
    assert_eq!(X.ncols(), weights.get("W").unwrap().nrows());
    assert_eq!(weights.get("B").unwrap().nrows(), weights.get("B").unwrap().ncols());

    let N: DMatrix<f32> = X * *weights.get("W").unwrap();
    let P: DMatrix<f32> = N + *weights.get("B").unwrap();
    let _loss: f32 = (y - P).map(|x| x*x).sum() / (y.nrows() as f32);
}


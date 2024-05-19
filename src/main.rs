
#![allow(non_snake_case, dead_code)]
mod derivs_better;
mod linear_regression;
use nalgebra::DMatrix;

use derivs_better::{matrix_fn_forward_sum, matrix_fn_backward_sum_1};
use linear_regression::{Weights};

use crate::linear_regression::initialise_weights;

// Make some function and compute the it's derivative

fn square(v: &[f32]) -> Vec<f32> {
    let u = v.iter().map(|x| x * x).collect();
    u
}

fn leaky_relu(v: &[f32]) -> Vec<f32> {
    let u = v.iter().map(|&x| if x > 0. { x } else { 0.1 * x}).collect();
    u
}

fn is_within_tolerance(a: &[f32], b: &[f32], tol: f32) -> bool {
    return a.iter().enumerate().all(|(i, x)| (x - b[i]).abs() <= tol)
}

fn sigmoidV2(x: f32) -> f32 {
    1f32 / (1f32 + (-x).exp())
}

fn main() {

    let X: DMatrix<f32> = DMatrix::from_vec(1, 3, vec![1., 2., 3.]);
    let W: DMatrix<f32> = DMatrix::from_vec(1, 3, vec![0.1, 0.2, 0.3]);
    println!("{:?}", &X * &W.transpose());

    let X= DMatrix::from_row_slice(3, 3, &[
        -1.57752816, -0.6664228 ,  0.63910406,
        -0.56152218,  0.73729959, -1.42307821,
        -1.44348429, -0.39128029,  0.1539322 
    ]);

    let W = DMatrix::from_row_slice(3, 2, &[
        0.75510818,  0.25562492,
        -0.56109271, -0.97504841,
        0.98098478, -0.95870776
    ]);

    println!("{}", matrix_fn_forward_sum(&X, &W, sigmoidV2));

    println!("{}", matrix_fn_backward_sum_1(&X, &W, sigmoidV2, 0.001));

    println!("Moving onto linear regression...");
    println!("Construct a struct of randomly initialised weights");

    let rand_weights: Weights = initialise_weights(10);
    println!("matrix weights: {}, intercept coef: {}", rand_weights.W, rand_weights.B);
    
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DMatrixView};
    use approx::relative_eq;

    use super::*;

    #[test]
    fn test_square_elements() {
        let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
        let test1 = square(&v);
        assert_eq!(test1, vec![1., 4., 9., 16., 25.]);
    }

    #[test]
    fn test_leaky_relu() {
        let v: Vec<f32> = vec![1., -2., 3., 4., 5.];
        let test1 = leaky_relu(&v);
        assert_eq!(test1, vec![1.0, -0.2, 3.0, 4., 5.]);
    }

    #[test]
    #[allow(unused)]
    fn test_fn_backward_sum() {
        let X= DMatrix::from_row_slice(3, 3, &[
            -1.57752816, -0.6664228 ,  0.63910406,
            -0.56152218,  0.73729959, -1.42307821,
            -1.44348429, -0.39128029,  0.1539322 
        ]);
    
        let W = DMatrix::from_row_slice(3, 2, &[
            0.75510818,  0.25562492,
            -0.56109271, -0.97504841,
            0.98098478, -0.95870776
        ]);

        
        let sol = DMatrix::from_row_slice(3, 3, &[
            0.2488747,  -0.37476832, 0.011204496, 
            0.12603968,  -0.27807757,  -0.13946879, 
            0.22991648,  -0.36621478, -0.022522569 
        ]);
        
        relative_eq!(matrix_fn_backward_sum_1(&X, &W, sigmoidV2, 0.001), sol);
    }

    #[test]
    fn test_matrix_slice() {
        let mat: DMatrix<f32> = DMatrix::from_row_slice(4, 4, &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);

        let view: DMatrixView<f32> = mat.view((1, 1), (2, 2));
        let expected_view: DMatrix<f32> = DMatrix::from_row_slice(2, 2, &[
            6.0, 7.0,
            10.0, 11.0
        ]);
        let notexpected_view: DMatrix<f32> = DMatrix::from_row_slice(2, 2, &[
            6.0, 7.0,
            10.0, 12.0
        ]);
        assert_eq!(view, expected_view);
        assert_ne!(view, notexpected_view);
    }
}

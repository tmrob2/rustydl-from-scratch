
mod derivative;
mod derivs_better;
use nalgebra::DMatrix;

use derivative::{hello_from_module_test, chain2, deriv, matrix_transpose, matrix_dot_product};
use derivs_better::{matrix_fn_forward_sum, matrix_fn_backward_sum_1};

// Make some function and compute the it's derivative

fn square(v: &[f32]) -> Vec<f32> {
    let u = v.iter().map(|x| x * x).collect();
    u
}

fn leaky_relu(v: &[f32]) -> Vec<f32> {
    let u = v.iter().map(|&x| if x > 0. { x } else { 0.1 * x}).collect();
    u
}

fn sigmoid(v: &[f32]) -> Vec<f32> {
    return v.iter().map(|x| 1f32 / (1f32 + (-x).exp())).collect()
}

fn is_within_tolerance(a: &[f32], b: &[f32], tol: f32) -> bool {
    return a.iter().enumerate().all(|(i, x)| (x - b[i]).abs() <= tol)
}

fn sigmoidV2(x: f32) -> f32 {
    1f32 / (1f32 + (-x).exp())
}

fn main() {
    hello_from_module_test();

    // Compute the derivative of f(x) = x^2
    // for the vector (1, 2, 3, 4)
    // df(x)/dx = 2.x
    // result vector should by (1, 4, 6, 8)
    let v: Vec<f32> = vec![1., 2., 3., 4.];
    println!("{:?}", deriv(square, &v, 0.001));

    let v: Vec<f32> = vec![1., 2., -3., 4.];
    println!("{:?}", deriv(leaky_relu, &v, 0.001));

    // Chain two functions
    let chain_result = chain2(square, sigmoid, &v, 0.001);
    println!("{:?}", chain_result);

    // Compute the matrix transpose
    println!("Computing the matrix transpose");
    let M: Vec<Vec<f32>> = vec![vec![1f32, 2f32, 3f32], vec![4f32, 5f32, 6f32]];
    let MT = matrix_transpose(&M);
    for r in 0..MT.len() {
        println!("{:?}", MT[r]);
    }

    println!("Computing the matrix multiplication of the matrices");
    let X: Vec<Vec<f32>> = vec![vec![-1.45813123,  0.26803587, -0.66724663]];
    let W: Vec<Vec<f32>> = vec![vec![-1.10984218], vec![0.6128276], vec![0.21915948]];
    // Compute the matrix multiplication of the matrices X, W
    println!("Number of rows, cols in X {}, {}; Number of rows, cols in W: {}, {}", 
        X.len(), X[0].len(), W.len(), W[0].len());
    let test: Vec<Vec<f32>> = matrix_dot_product(&X, &W);
    for r in 0..test.len() {
        println!("{:?}", test[r]);
    }

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
    
}

mod tests {
    use nalgebra::DMatrix;

    use self::derivative::matrix_dot_product;

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
    fn test_deriv_square() {
        let v: Vec<f32> = vec![1., 2., 3., 4.];
        let test = deriv(square, &v, 0.001);
        assert!(is_within_tolerance(&test, &vec![2., 4., 6., 8.], 0.01));
    }

    #[test]
    fn test_dot_product() {
        let X: Vec<Vec<f32>> = vec![vec![-1.45813123,  0.26803587, -0.66724663]];
        let W: Vec<Vec<f32>> = vec![vec![-1.10984218, 0.6128276, 0.21915948]];
        // Compute the matrix multiplication of the matrices X, W
        let test = matrix_dot_product(&X, &W);
    }
}

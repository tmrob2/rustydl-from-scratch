#![allow(non_snake_case, dead_code)]
mod derivs_better;
mod linear_regression;
mod manual_nn;
mod metrics;

use pyo3::prelude::*;
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use linear_regression::LinearRegression; 

/// Formats the sum of two numbers as string.
#[pymodule]
fn rust_dl<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // write a wrapper for linear regression
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray_bound(py)
    }

    // Linear Regression Class
    m.add_class::<LinearRegression>()?;

    Ok(())
}


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


#[cfg(test)]
mod tests {
    use ndarray::{Array2, arr2, s};
    use approx::abs_diff_eq;

    use super::*;
    use derivs_better::matrix_fn_backward_sum_1;

    #[test]
    fn test_square_elements() {
        let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
        let test1 = square(&v);
        assert_eq!(test1, vec![1., 4., 9., 16., 25.]);
    }

    #[test]
    fn dot_product() {
        let X: Array2<f32> = Array2::from_shape_vec((1, 3), vec![1., 2., 3.]).unwrap();
        let W: Array2<f32> = Array2::from_shape_vec((1, 3), vec![0.1, 0.2, 0.3]).unwrap();
        let prod: Array2<f32> = X.dot(&W.t());
        let sol: Array2<f32> = arr2(&[[1.4]]);
        prod.iter().zip(sol.iter()).all(|(x, y)| abs_diff_eq!(x, y, epsilon=1e-5));
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
        let X: Array2<f32>= arr2(&[
            [-1.57752816, -0.6664228 ,  0.63910406],
            [-0.56152218,  0.73729959, -1.42307821],
            [-1.44348429, -0.39128029,  0.1539322] 
        ]);
    
        let W: Array2<f32> = arr2(&[
            [0.75510818,  0.25562492],
            [-0.56109271, -0.97504841],
            [0.98098478, -0.95870776]
        ]);

        
        let sol: Array2<f32> = arr2(&[
            [0.2488747,  -0.37476832, 0.011204496], 
            [0.12603968,  -0.27807757,  -0.13946879], 
            [0.22991648,  -0.36621478, -0.022522569] 
        ]);
        
        let computed: Array2<f32> = matrix_fn_backward_sum_1(&X, &W, sigmoidV2, 0.001);
        
        computed.iter().zip(sol.iter()).all(|(x, y)| abs_diff_eq!(x, y));
        //sol.abs_diff_eq(computed, 1e-5);
    }

    #[test]
    fn test_matrix_slice() {
        let mat: Array2<f32> = arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        let view = mat.slice(s![1..3, 1..3]);
        let expected_view = arr2(&[
            [6.0, 7.0],
            [10.0, 11.0]
        ]);
        let notexpected_view = arr2(&[
            [6.0, 7.0],
            [10.0, 12.0]
        ]);
        assert_eq!(view, expected_view);
        assert_ne!(view, notexpected_view);
    }
}
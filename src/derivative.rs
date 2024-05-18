use num_traits::Float;

pub fn hello_from_module_test() {
    println!("hello from the other module");
}

pub fn deriv<F>(f: F, a: &[f32], delta: f32) -> Vec<f32>
where F: Fn(&[f32]) -> Vec<f32> {
    // For each value in a add delta to it
    let aplus_delta: Vec<f32> = a.iter().map(|x| x + delta).collect();
    let asub_delta: Vec<f32> = a.iter().map(|x| x - delta).collect();

    let z1: Vec<f32> = f(&aplus_delta);
    let z2: Vec<f32> = f(&asub_delta);
    let z3: Vec<f32> = z1.iter().enumerate().map(|(i, x)| x - z2[i]).collect();
    let dfa_du: Vec<f32> = z3.iter().map(|x| x / (2.0 * delta)).collect();
    dfa_du
}

pub fn chain2<F1, F2>(f1: F1, f2: F2, x: &[f32], delta: f32) -> Vec<f32> 
where F1: Fn(&[f32]) -> Vec<f32>, 
      F2: Fn(&[f32]) -> Vec<f32>{
    let df1du = deriv(&f1, x, delta);
    let df2du = deriv(f2, &f1(x), delta);
    let result = df1du.iter().enumerate().map(|(i, x)| df2du[i] * x).collect();
    result
}

pub fn multiple_inputs_add<F>(x: &[f32], y: &[f32], f: F) -> Vec<f32> 
where F: Fn(&[f32]) -> Vec<f32> {
    let a: Vec<f32> = x.iter().zip(y.iter()).map(|(x_, y_)| x_ + y_).collect();
    f(&a)
}

pub fn multiple_inputs_add_backward<F>(x: &[f32], y: &[f32], f: F) -> (Vec<f32>, Vec<f32>)
where F: Fn(&[f32]) -> Vec<f32> {
    // compute the foward phase
    let a: Vec<f32> = x.iter().zip(y.iter()).map(|(x_, y_)| x_ + y_).collect();
    let ds_da: Vec<f32> = deriv(f, &a, 0.001);
    let ds_da_da_dx: Vec<f32> = ds_da.iter().map(|x| x * 1.0).collect();
    return (ds_da, ds_da_da_dx)
}

fn dot_product(x: &[f32], w: &[f32]) -> f32 {
    x.iter().zip(w.iter()).map(|(x_, w_)| x_ * w_).sum()
}

pub fn matrix_dot_product(X: &[Vec<f32>], W: &[Vec<f32>]) -> Vec<Vec<f32>> {
    // This function makes the assumption that W is already in transpose form
    // How many rows are there in X
    let r: usize = X.len();
    let c: usize = W[0].len();

    let mut XW: Vec<Vec<f32>> = vec![vec![0.; c]; r];

    for ii in 0..r {
        for jj in 0..c {
            // compute the dot product of the matrix X row and W col
            let result = dot_product(&X[ii], &W[jj]);
            XW[jj][ii] = result;
        }
    }
    XW
}

pub fn matrix_transpose(M: &[Vec<f32>]) -> Vec<Vec<f32>> {
    // Define a new matrix
    let mut T: Vec<Vec<f32>> = vec![vec![0.; M.len()]; M[0].len()];
    for r in 0..M.len() {
        for c in 0..M[0].len() {
            T[c][r] = M[r][c];
        }
    }
    T
}

pub fn matrix_function_backward_1<F>(X: &[Vec<f32>], W: &[Vec<f32>], sigma: F)
where F: Fn(&[f32]) -> &[f32] {
    // Compute the dot product between two matrices

    let N = matrix_dot_product(X, W);
}
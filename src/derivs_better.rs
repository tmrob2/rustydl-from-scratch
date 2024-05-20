/**
 * This module uses nalgebra as a backbone instead of Vec<Vec<f32>>
 * 
 * The righ matrix to use is the Smatrix which is a statically sized matrix which we must know it's size
 */

extern crate blas_src;

use ndarray::Array2;

#[allow(non_snake_case)]
pub fn deriv<F>(f: F, a: &Array2<f32>, delta: f32) -> Array2<f32> 
where F: Fn(f32) -> f32 {
   // apply the function f t
   // add a small f32 value to each element in the matrix
   let mut aplus_delta: Array2<f32> = a + delta;
   let mut aminus_delta: Array2<f32> = a - delta;
   //println!("Derivatives: \n, A + delta: {}, A - delta: {}", aplus_delta, aminus_delta);

   // it's probably better if the function just maps a floating point number to a new number
   // and then we do the matrix iteration in this function
   // also we want to edit the matrices in place so this we don't have to create new matrices
   aplus_delta = aplus_delta.map(|x| f(*x));
   aminus_delta = aminus_delta.map(|x| f(*x));
   //println!("Application of fn to A\nA + delta: {}, A - delta: {}", aplus_delta, aminus_delta);
   // because we are subtracting one matrix from another we can use a blas function to do this
   // I believe axpy is appropriate here
   (aplus_delta - aminus_delta) * 1f32 / (2f32 * delta)
}

#[allow(non_snake_case, dead_code)]
pub fn chain_two_fn<F1, F2>(f1: F1, f2: F2, input: &Array2<f32>) -> Array2<f32>
where F1: Fn(&Array2<f32>) -> Array2<f32>,
      F2: Fn(&Array2<f32>) -> Array2<f32> {
   f2(&f1(input))
}

#[allow(non_snake_case, dead_code)]
pub fn chain_deriv_2<F1, F2>(f1: F1, f2: F2, input: &Array2<f32>, delta: f32) -> Array2<f32> 
where F1: Fn(f32) -> f32,
      F2: Fn(f32) -> f32 {
   let f1_of_x: Array2<f32> = input.map(|x| f1(*x));
   let df1dx: Array2<f32> = deriv(f1, input, delta);
   let df2du: Array2<f32> = deriv(f2, &f1_of_x, delta);
   return df1dx * df2du
}

#[allow(non_snake_case, dead_code)]
pub fn chain_deriv_3<F1, F2, F3>(f1: F1, f2: F2, f3: F3, input: &Array2<f32>, delta: f32) -> Array2<f32> 
where F1: Fn(f32) -> f32,
      F2: Fn(f32) -> f32,
      F3: Fn(f32) -> f32 {
   let f1_of_x: Array2<f32> = input.map(|x| f1(*x));
   let f2_of_x: Array2<f32> = f1_of_x.map(|x| f2(*x));
   let df3du: Array2<f32> = deriv(f3, &f2_of_x, delta);
   let df2du: Array2<f32> = deriv(f2, &f1_of_x, delta);
   let df1dx: Array2<f32> = deriv(f1, &input, delta);
   df1dx * df2du * df3du
}

#[allow(non_snake_case, dead_code)]
pub fn multiple_inputs_add<F>(x: &Array2<f32>, y: &Array2<f32>, f: F) -> Array2<f32> 
where F: Fn(f32) -> f32 {
   let z: Array2<f32> = x + y;
   z.map(|x| f(*x))
}

#[allow(non_snake_case, dead_code)]
pub fn multiple_inputs_add_backward<F>(x: &Array2<f32>, y: &Array2<f32>, f: F, delta: f32) -> Array2<f32> 
where F: Fn(f32) -> f32 {
   // Compute the forward pass
   let z: Array2<f32> = x+y;
   // Compute the derivatives
   deriv(f, &z, delta)
}

#[allow(non_snake_case, dead_code)]
pub fn mat_mul_forward<F>(x: &Array2<f32>, w: &Array2<f32>, f: F) -> Array2<f32>
where F: Fn(f32) -> f32 {
   let z: Array2<f32> = x.dot(&w.t());
   // apply the function to every element of the matrix
   z.map(|x| f(*x))
}

#[allow(non_snake_case, dead_code)]
pub fn matrix_fn_backward_1<F>(X: &Array2<f32>, W: &Array2<f32>, f: F, delta: f32) -> Array2<f32> 
where F: Fn(f32) -> f32 {
   let fRes: Array2<f32> = mat_mul_forward(X, W, &f);
   // do the backward calculation
   // TODO Make a decision here whether we make the input to deriv mutable, and don't allocate to
   //  a new memory address. 
   let dSdN: Array2<f32> = deriv(f, &fRes, delta);
   let dNdX = W.t();
   return dSdN.dot(&dNdX);
}

#[allow(non_snake_case, dead_code)]
pub fn matrix_fn_forward_sum<F>(x: &Array2<f32>, w: &Array2<f32>, f: F) -> f32
where F: Fn(f32) -> f32 {
   let n: Array2<f32> = x.dot(w);
   let s: Array2<f32> = n.map(|x| f(*x));
   s.sum()
}

#[allow(non_snake_case, dead_code)]
pub fn matrix_fn_backward_sum_1<F>(x: &Array2<f32>, w: &Array2<f32>, f: F, delta: f32) -> Array2<f32> 
where F: Fn(f32) -> f32 {
   //println!("X: {}", x);
   //println!("w: {}", w);
   let N: Array2<f32> = x.dot(w);
   //println!("N: {}", N);
   //et S: Array2<f32> = N.map(|x| f(x));
   //println!("N: {}", N);
   //let L = S.sum();

   //let dLdS: Array2<f32> = Array2::from_element(S.nrows(), S.ncols(), 1f32);
   //println!("dLdS: {}", dLdS);

   let dSdN: Array2<f32> = deriv(f, &N, delta);
   
   //let dLdN: Array2<f32>  = dLdS * dSdN;
   //println!("dSdN: {}", dSdN);
   
   //let dNdX = w.t();
   
   let dLdX: Array2<f32> = dSdN.dot(&w.t());

   dLdX
}


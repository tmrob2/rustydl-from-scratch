/**
 * This module uses nalgebra as a backbone instead of Vec<Vec<f32>>
 * 
 * The righ matrix to use is the Smatrix which is a statically sized matrix which we must know it's size
 */

 use nalgebra::DMatrix;


pub fn deriv<F>(f: F, a: &DMatrix<f32>, delta: f32) -> DMatrix<f32> 
where F: Fn(f32) -> f32 {
   // apply the function f t
   // add a small f32 value to each element in the matrix
   let mut aplus_delta: DMatrix<f32>  = a.add_scalar(delta);
   let mut aminus_delta: DMatrix<f32>   = a.add_scalar(delta);

   // it's probably better if the function just maps a floating point number to a new number
   // and then we do the matrix iteration in this function
   // also we want to edit the matrices in place so this we don't have to create new matrices
   aplus_delta.iter_mut().map(|x| f(*x)).for_each(drop);
   aminus_delta.iter_mut().map(|x| f(*x)).for_each(drop);
   // because we are subtracting one matrix from another we can use a blas function to do this
   // I believe axpy is appropriate here
   (aplus_delta - aminus_delta) * 1f32 / (2f32 * delta)
}

pub fn chain_two_fn<F1, F2>(f1: F1, f2: F2, input: &DMatrix<f32>, delta: f32) -> DMatrix<f32>
where F1: Fn(&DMatrix<f32>) -> DMatrix<f32>,
      F2: Fn(&DMatrix<f32>) -> DMatrix<f32> {
   f2(&f1(input))
}

pub fn chain_deriv_2<F1, F2>(f1: F1, f2: F2, input: &DMatrix<f32>, delta: f32) -> DMatrix<f32> 
where F1: Fn(f32) -> f32,
      F2: Fn(f32) -> f32 {
   let f1_of_x: DMatrix<f32> = input.map(|x| f1(x));
   let df1dx: DMatrix<f32> = deriv(f1, input, delta);
   let df2du: DMatrix<f32> = deriv(f2, &f1_of_x, delta);
   return df1dx * df2du
}

pub fn chain_deriv_3<F1, F2, F3>(f1: F1, f2: F2, f3: F3, input: &DMatrix<f32>, delta: f32) -> DMatrix<f32> 
where F1: Fn(f32) -> f32,
      F2: Fn(f32) -> f32,
      F3: Fn(f32) -> f32 {
   let f1_of_x: DMatrix<f32> = input.map(|x| f1(x));
   let f2_of_x: DMatrix<f32> = f1_of_x.map(|x| f2(x));
   let df3du: DMatrix<f32> = deriv(f3, &f2_of_x, delta);
   let df2du: DMatrix<f32> = deriv(f2, &f1_of_x, delta);
   let df1dx: DMatrix<f32> = deriv(f1, &input, delta);
   df1dx * df2du * df3du
}

pub fn multiple_inputs_add<F>(x: &DMatrix<f32>, y: &DMatrix<f32>, f: F) -> DMatrix<f32> 
where F: Fn(f32) -> f32 {
   let z: DMatrix<f32> = x + y;
   z.map(|x| f(x))
}

pub fn multiple_inputs_add_backward<F>(x: &DMatrix<f32>, y: &DMatrix<f32>, f: F, delta: f32) -> DMatrix<f32> 
where F: Fn(f32) -> f32 {
   // Compute the forward pass
   let z: DMatrix<f32> = x+y;
   // Compute the derivatives
   deriv(f, x, delta)
}

pub fn mat_mul_forward<F>(x: &DMatrix<f32>, w: &DMatrix<f32>, f: F) -> DMatrix<f32>
where F: Fn(f32) -> f32 {
   let z: DMatrix<f32> = x * w.transpose();
   // apply the function to every element of the matrix
   z.map(|x| f(x))
}

pub fn matrix_fn_backward_1<F>(X: &DMatrix<f32>, W: &DMatrix<f32>, f: F, delta: f32) -> DMatrix<f32> 
where F: Fn(f32) -> f32 {
   let fRes: DMatrix<f32> = mat_mul_forward(X, W, &f);
   // do the backward calculation
   // TODO Make a decision here whether we make the input to deriv mutable, and don't allocate to
   //  a new memory address. 
   let dSdN: DMatrix<f32> = deriv(f, &fRes, delta);
   let dNdX: DMatrix<f32> = W.transpose();
   return dSdN * dNdX;
}

pub fn matrix_fn_forward_sum<F>(x: &DMatrix<f32>, w: &DMatrix<f32>, f: F) -> f32
where F: Fn(f32) -> f32 {
   let n: DMatrix<f32> = x * w;
   let s: DMatrix<f32> = n.map(|x| f(x));
   s.sum()
}

pub fn matrix_fn_backward_sum_1<F>(x: &DMatrix<f32>, w: &DMatrix<f32>, f: F, delta: f32) -> DMatrix<f32> 
where F: Fn(f32) -> f32 {
   let N: DMatrix<f32> = x * w;
   let S: DMatrix<f32> = N.map(|x| f(x));
   println!("{}", N);
   let L = S.sum();

   let dLdS: DMatrix<f32> = DMatrix::from_element(S.nrows(), S.ncols(), 1f32);
   println!("dLdS: {}", dLdS);

   let dSdN: DMatrix<f32> = deriv(f, &N, delta);
   
   //let dLdN: DMatrix<f32>  = dLdS * dSdN;
   println!("dSdN: {}", dSdN);
   
   let dNdX: DMatrix<f32> = w.transpose();


   let dLdX: DMatrix<f32> = dSdN * dNdX;

   dLdX
}


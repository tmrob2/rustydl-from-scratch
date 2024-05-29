// This module implements the metrics for statistical learning
use numpy::ndarray::ArrayView2;

pub fn r2_score(pred: ArrayView2<f32>, observed: ArrayView2<f32>) -> f32 {
    // R^2 = 1 - SS_res / SS_tot or sum of residual squares divided by sum of total squares
    let ss_res: f32 = (&observed - &pred).map(|x| x * x).sum();
    let ymean: f32 = observed.mean().unwrap();
    let ss_tot: f32 = (&observed - ymean).map(|x| x * x).sum();
    1f32 - ss_res / ss_tot
}
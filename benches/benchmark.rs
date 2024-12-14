//! Suite of simple benchmarks focused on the `Polynomial::eval()` method.
//! The suite covers floating point (`f64`) and integer values (`i64`),
//! in addition to arbitrary precision arithmetic (`BigRational`).
//!
//! For each data type there are two helper functions: one that is used to construct
//! a representative set of polynomials for that data type, and a second to actually
//! evaluate a set (`array`) of polynomials drawn from that set. The final "benchmarked
//! function" (and thus time) for each type is the sum across all evaluating every
//! polynomial in each set for a representative set of times.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use iter_num_tools::lin_space;
use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::BigRational;
use polynomial::Polynomial;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn create_rng_scalar_polynomial<R, D>(num_samples: usize, rng: &mut R, dist: &D) -> Polynomial<f64>
where
    R: Rng,
    D: Distribution<f64>,
{
    Polynomial::new((0..num_samples).map(|_| dist.sample(rng)).collect())
}

fn eval_scalar_polynomial(num_samples: usize, poly_arr: &[Polynomial<f64>; 3]) {
    let queries = lin_space(-5.0..=5.0, num_samples);
    for query in queries {
        for poly in poly_arr {
            black_box(poly.eval(query));
        }
    }
}

fn create_rng_complex_polynomial<R, D>(
    num_samples: usize,
    rng: &mut R,
    dist: &D,
) -> Polynomial<Complex<f64>>
where
    R: Rng,
    D: Distribution<f64>,
{
    Polynomial::new(
        (0..num_samples)
            .map(|_| Complex::new(dist.sample(rng), dist.sample(rng)))
            .collect(),
    )
}

fn eval_complex_polynomial(sqrt_num_samples: usize, poly_arr: &[Polynomial<Complex<f64>>; 3]) {
    let real_iter = lin_space(-5.0..=5.0, sqrt_num_samples);
    for real in real_iter {
        let imag_iter = lin_space(-5.0..=5.0, sqrt_num_samples);
        for imag in imag_iter {
            let value = Complex::new(real, imag);
            for poly in poly_arr {
                black_box(poly.eval(value));
            }
        }
    }
}

fn create_rng_big_rational_polynomial<R, D>(
    num_samples: usize,
    rng: &mut R,
    dist: &D,
) -> Polynomial<BigRational>
where
    R: Rng,
    D: Distribution<i64>,
{
    Polynomial::new(
        (0..num_samples)
            .map(|_| {
                let num = BigInt::from(dist.sample(rng)) * BigInt::from(dist.sample(rng));
                let den = BigInt::from(dist.sample(rng)) * BigInt::from(dist.sample(rng));
                BigRational::new(num, den)
            })
            .collect(),
    )
}

fn eval_big_rational_polynomial(num_samples: usize, poly_arr: &[Polynomial<BigRational>; 3]) {
    let num_iter = lin_space(-500_000..=500_000, num_samples);
    let den = BigInt::from(100_000);
    for num in num_iter {
        let query = BigRational::new(BigInt::from(num), den.clone());
        for poly in poly_arr {
            black_box(poly.eval(query.clone()));
        }
    }
}

fn benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(12345);
    let f64_dist = Uniform::from(-9.0..9.0);

    {
        let pole_test_set: [_; 3] =
            std::array::from_fn(|i| create_rng_scalar_polynomial(i + 3, &mut rng, &f64_dist));
        c.bench_function("eval_scalar_polynomial", |b| {
            b.iter(|| eval_scalar_polynomial(5000, &pole_test_set))
        });
    }

    {
        let pole_test_set: [_; 3] =
            std::array::from_fn(|i| create_rng_complex_polynomial(i + 3, &mut rng, &f64_dist));
        c.bench_function("eval_complex_polynomial", |b| {
            b.iter(|| eval_complex_polynomial(72, &pole_test_set))
        });
    }

    {
        let i64_dist = Uniform::from(i64::MIN..i64::MAX);
        let pole_test_set: [_; 3] =
            std::array::from_fn(|i| create_rng_big_rational_polynomial(i + 3, &mut rng, &i64_dist));
        c.bench_function("eval_big_rational_polynomial", |b| {
            b.iter(|| eval_big_rational_polynomial(200, &pole_test_set))
        });
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);

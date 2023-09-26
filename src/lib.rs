//! A library for manipulating polynomials.
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(bad_style)]
#![warn(missing_docs)]
#![warn(trivial_casts)]
#![warn(trivial_numeric_casts)]
#![warn(unused)]
#![warn(unused_extern_crates)]
#![warn(unused_import_braces)]
#![warn(unused_qualifications)]
#![warn(unused_results)]

use core::ops::{Add, Div, Mul, Neg, Sub};
use core::{cmp, fmt};
use num_traits::{FromPrimitive, One, Zero};

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// A polynomial.
#[derive(Eq, PartialEq, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Polynomial<T> {
    data: Vec<T>,
}

impl<T: Zero> Polynomial<T> {
    /// Creates a new `Polynomial` from a `Vec` of coefficients.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial::Polynomial;
    /// let poly = Polynomial::new(vec![1, 2, 3]);
    /// assert_eq!("1+2*x+3*x^2", poly.pretty("x"));
    /// ```
    #[inline]
    pub fn new(mut data: Vec<T>) -> Self {
        while let Some(true) = data.last().map(|x| x.is_zero()) {
            let _ = data.pop();
        }
        Self { data }
    }
}

impl<T> Polynomial<T>
where
    T: One + Zero + Clone + Neg<Output = T> + Div<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    /// Creates the [Lagrange polynomial] that fits a number of points.
    ///
    /// [Lagrange polynomial]: https://en.wikipedia.org/wiki/Lagrange_polynomial
    ///
    /// Returns `None` if any two x-coordinates are the same.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial::Polynomial;
    /// let poly = Polynomial::lagrange(&[1, 2, 3], &[10, 40, 90]).unwrap();
    /// println!("{}", poly.pretty("x"));
    /// assert_eq!("10*x^2", poly.pretty("x"));
    /// ```
    #[inline]
    pub fn lagrange(xs: &[T], ys: &[T]) -> Option<Self> {
        let mut res = Polynomial::new(vec![Zero::zero()]);
        for ((i, x), y) in (0..).zip(xs.iter()).zip(ys.iter()) {
            let mut p: Polynomial<T> = Polynomial::new(vec![T::one()]);
            let mut denom = T::one();
            for (j, x2) in (0..).zip(xs.iter()) {
                if i != j {
                    p = p * &Polynomial::new(vec![-x2.clone(), T::one()]);
                    let diff = x.clone() - x2.clone();
                    if diff.is_zero() {
                        return None;
                    }
                    denom = denom * diff;
                }
            }
            let scalar = y.clone() / denom;
            res = res + p * &Polynomial::<T>::new(vec![scalar]);
        }
        Some(res)
    }
}

impl<T> Polynomial<T>
where
    T: One
        + Zero
        + Clone
        + Neg<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + FromPrimitive,
{
    /// [Chebyshev approximation] fits a function to a polynomial over a range of values.
    ///
    /// [Chebyshev approximation]: https://en.wikipedia.org/wiki/Approximation_theory#Chebyshev_approximation
    ///
    /// This attempts to minimize the maximum error.
    ///
    /// Retrurns `None` if `n < 1` or `xmin >= xmax`.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial::Polynomial;
    /// use std::f64::consts::PI;
    /// let p = Polynomial::chebyshev(&f64::sin, 7, -PI/4., PI/4.).unwrap();
    /// assert!((p.eval(0.) - (0.0_f64).sin()).abs() < 0.0001);
    /// assert!((p.eval(0.1) - (0.1_f64).sin()).abs() < 0.0001);
    /// assert!((p.eval(-0.1) - (-0.1_f64).sin()).abs() < 0.0001);
    /// ```
    #[inline]
    pub fn chebyshev<F: Fn(T) -> T>(f: &F, n: usize, xmin: f64, xmax: f64) -> Option<Self> {
        if n < 1 || xmin >= xmax {
            return None;
        }

        let mut xs = Vec::new();
        for i in 0..n {
            use core::f64::consts::PI;
            let x = T::from_f64(
                (xmax + xmin) * 0.5
                    + (xmin - xmax) * 0.5 * ((2 * i + 1) as f64 * PI / (2 * n) as f64).cos(),
            )
            .unwrap();
            xs.push(x);
        }

        let ys: Vec<T> = xs.iter().map(|x| f(x.clone())).collect();
        Polynomial::lagrange(&xs[0..], &ys[0..])
    }
}

impl<T: Zero + Mul<Output = T> + Clone> Polynomial<T> {
    /// Evaluates the polynomial at a point.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial::Polynomial;
    /// let poly = Polynomial::new(vec![1, 2, 3]);
    /// assert_eq!(1, poly.eval(0));
    /// assert_eq!(6, poly.eval(1));
    /// assert_eq!(17, poly.eval(2));
    /// ```
    #[inline]
    pub fn eval(&self, x: T) -> T {
        let mut result: T = Zero::zero();
        for n in self.data.iter().rev() {
            result = n.clone() + result * x.clone();
        }
        result
    }
}

impl<T> Polynomial<T> {
    /// Gets the slice of internal data.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }
}

impl<T> Polynomial<T>
where
    T: Zero + One + Eq + Neg<Output = T> + Ord + fmt::Display + Clone,
{
    /// Pretty prints the polynomial.
    pub fn pretty(&self, x: &str) -> String {
        if self.is_zero() {
            return "0".to_string();
        }

        let one = One::one();
        let mut s = Vec::new();
        for (i, n) in self.data.iter().enumerate() {
            // output n*x^i / -n*x^i
            if n.is_zero() {
                continue;
            }

            let term = if i.is_zero() {
                n.to_string()
            } else if i == 1 {
                if (*n) == one {
                    x.to_string()
                } else if (*n) == -one.clone() {
                    format!("-{}", x)
                } else {
                    format!("{}*{}", n, x)
                }
            } else if (*n) == one {
                format!("{}^{}", x, i)
            } else if (*n) == -one.clone() {
                format!("-{}^{}", x, i)
            } else {
                format!("{}*{}^{}", n, x, i)
            };

            if !s.is_empty() && (*n) > Zero::zero() {
                s.push("+".to_string());
            }
            s.push(term);
        }

        s.concat()
    }
}

impl<T> Neg for Polynomial<T>
where
    T: Neg + Zero + Clone,
    <T as Neg>::Output: Zero,
{
    type Output = Polynomial<<T as Neg>::Output>;

    #[inline]
    fn neg(self) -> Polynomial<<T as Neg>::Output> {
        -&self
    }
}

impl<'a, T> Neg for &'a Polynomial<T>
where
    T: Neg + Zero + Clone,
    <T as Neg>::Output: Zero,
{
    type Output = Polynomial<<T as Neg>::Output>;

    #[inline]
    fn neg(self) -> Polynomial<<T as Neg>::Output> {
        Polynomial::new(self.data.iter().map(|x| -x.clone()).collect())
    }
}

macro_rules! forward_val_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<Lhs, Rhs> $imp<Polynomial<Rhs>> for Polynomial<Lhs>
        where
            Lhs: Zero + $imp<Rhs> + Clone,
            Rhs: Zero + Clone,
            <Lhs as $imp<Rhs>>::Output: Zero,
        {
            type Output = Polynomial<<Lhs as $imp<Rhs>>::Output>;

            #[inline]
            fn $method(self, other: Polynomial<Rhs>) -> Polynomial<<Lhs as $imp<Rhs>>::Output> {
                (&self).$method(&other)
            }
        }
    };
}

macro_rules! forward_ref_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, Lhs, Rhs> $imp<Polynomial<Rhs>> for &'a Polynomial<Lhs>
        where
            Lhs: Zero + $imp<Rhs> + Clone,
            Rhs: Zero + Clone,
            <Lhs as $imp<Rhs>>::Output: Zero,
        {
            type Output = Polynomial<<Lhs as $imp<Rhs>>::Output>;

            #[inline]
            fn $method(self, other: Polynomial<Rhs>) -> Polynomial<<Lhs as $imp<Rhs>>::Output> {
                self.$method(&other)
            }
        }
    };
}

macro_rules! forward_val_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, Lhs, Rhs> $imp<&'a Polynomial<Rhs>> for Polynomial<Lhs>
        where
            Lhs: Zero + $imp<Rhs> + Clone,
            Rhs: Zero + Clone,
            <Lhs as $imp<Rhs>>::Output: Zero,
        {
            type Output = Polynomial<<Lhs as $imp<Rhs>>::Output>;

            #[inline]
            fn $method(self, other: &Polynomial<Rhs>) -> Polynomial<<Lhs as $imp<Rhs>>::Output> {
                (&self).$method(other)
            }
        }
    };
}

macro_rules! forward_all_binop {
    (impl $imp:ident, $method:ident) => {
        forward_val_val_binop!(impl $imp, $method);
        forward_ref_val_binop!(impl $imp, $method);
        forward_val_ref_binop!(impl $imp, $method);
    };
}

forward_all_binop!(impl Add, add);

impl<'a, 'b, Lhs, Rhs> Add<&'b Polynomial<Rhs>> for &'a Polynomial<Lhs>
where
    Lhs: Zero + Add<Rhs> + Clone,
    Rhs: Zero + Clone,
    <Lhs as Add<Rhs>>::Output: Zero,
{
    type Output = Polynomial<<Lhs as Add<Rhs>>::Output>;

    fn add(self, other: &Polynomial<Rhs>) -> Polynomial<<Lhs as Add<Rhs>>::Output> {
        let max_len = cmp::max(self.data.len(), other.data.len());
        let min_len = cmp::min(self.data.len(), other.data.len());

        let mut sum = Vec::with_capacity(max_len);
        for i in 0..min_len {
            sum.push(self.data[i].clone() + other.data[i].clone());
        }

        if self.data.len() <= other.data.len() {
            for i in min_len..max_len {
                sum.push(num_traits::zero::<Lhs>() + other.data[i].clone());
            }
        } else {
            for i in min_len..max_len {
                sum.push(self.data[i].clone() + num_traits::zero::<Rhs>());
            }
        }

        Polynomial::new(sum)
    }
}

forward_all_binop!(impl Sub, sub);

impl<'a, 'b, Lhs, Rhs> Sub<&'b Polynomial<Rhs>> for &'a Polynomial<Lhs>
where
    Lhs: Zero + Sub<Rhs> + Clone,
    Rhs: Zero + Clone,
    <Lhs as Sub<Rhs>>::Output: Zero,
{
    type Output = Polynomial<<Lhs as Sub<Rhs>>::Output>;

    fn sub(self, other: &Polynomial<Rhs>) -> Polynomial<<Lhs as Sub<Rhs>>::Output> {
        let min_len = cmp::min(self.data.len(), other.data.len());
        let max_len = cmp::max(self.data.len(), other.data.len());

        let mut sub = Vec::with_capacity(max_len);
        for i in 0..min_len {
            sub.push(self.data[i].clone() - other.data[i].clone());
        }
        if self.data.len() <= other.data.len() {
            for i in min_len..max_len {
                sub.push(num_traits::zero::<Lhs>() - other.data[i].clone())
            }
        } else {
            for i in min_len..max_len {
                sub.push(self.data[i].clone() - num_traits::zero::<Rhs>())
            }
        }
        Polynomial::new(sub)
    }
}

forward_all_binop!(impl Mul, mul);

impl<'a, 'b, Lhs, Rhs> Mul<&'b Polynomial<Rhs>> for &'a Polynomial<Lhs>
where
    Lhs: Zero + Mul<Rhs> + Clone,
    Rhs: Zero + Clone,
    <Lhs as Mul<Rhs>>::Output: Zero,
{
    type Output = Polynomial<<Lhs as Mul<Rhs>>::Output>;

    fn mul(self, other: &Polynomial<Rhs>) -> Polynomial<<Lhs as Mul<Rhs>>::Output> {
        if self.is_zero() || other.is_zero() {
            return Polynomial::new(vec![]);
        }

        let slen = self.data.len();
        let olen = other.data.len();
        let prod = (0..slen + olen - 1)
            .map(|i| {
                let mut p = num_traits::zero::<<Lhs as Mul<Rhs>>::Output>();
                let kstart = cmp::max(olen, i + 1) - olen;
                let kend = cmp::min(slen, i + 1);
                for k in kstart..kend {
                    p = p + self.data[k].clone() * other.data[i - k].clone();
                }
                p
            })
            .collect();
        Polynomial::new(prod)
    }
}

impl<T: Zero + Clone> Zero for Polynomial<T> {
    #[inline]
    fn zero() -> Self {
        Self { data: vec![] }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Zero + One + Clone> One for Polynomial<T> {
    #[inline]
    fn one() -> Self {
        Self {
            data: vec![One::one()],
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    extern crate alloc;

    #[cfg(not(feature = "std"))]
    use alloc::{string::ToString, vec, vec::Vec};

    use super::Polynomial;

    #[test]
    fn new() {
        fn check(dst: Vec<i32>, src: Vec<i32>) {
            assert_eq!(dst, Polynomial::new(src).data);
        }
        check(vec![1, 2, 3], vec![1, 2, 3]);
        check(vec![1, 2, 3], vec![1, 2, 3, 0, 0]);
        check(vec![], vec![0, 0, 0]);
    }

    #[test]
    fn neg_add_sub() {
        fn check(a: &[i32], b: &[i32], c: &[i32]) {
            fn check_eq(a: &Polynomial<i32>, b: &Polynomial<i32>) {
                assert_eq!(*a, *b);
                assert_eq!(-a, -b);
            }
            fn check_add(sum: &Polynomial<i32>, a: &Polynomial<i32>, b: &Polynomial<i32>) {
                check_eq(sum, &(a + b));
                check_eq(sum, &(b + a));
            }
            fn check_sub(sum: &Polynomial<i32>, a: &Polynomial<i32>, b: &Polynomial<i32>) {
                check_eq(a, &(sum - b));
                check_eq(b, &(sum - a));
            }

            let a = &Polynomial::new(a.to_vec());
            let b = &Polynomial::new(b.to_vec());
            let c = &Polynomial::new(c.to_vec());
            check_add(c, a, b);
            check_add(&(-c), &(-a), &(-b));
            check_sub(c, a, b);
            check_sub(&(-c), &(-a), &(-b));
        }
        check(&[], &[], &[]);
        check(&[], &[1], &[1]);
        check(&[1], &[1], &[2]);
        check(&[1, 0, 1], &[1], &[2, 0, 1]);
        check(&[1, 0, -1], &[-1, 0, 1], &[]);
    }

    #[test]
    fn mul() {
        fn check(a: &[i32], b: &[i32], c: &[i32]) {
            let a = Polynomial::new(a.to_vec());
            let b = Polynomial::new(b.to_vec());
            let c = Polynomial::new(c.to_vec());
            assert_eq!(c, &a * &b);
            assert_eq!(c, &b * &a);
        }
        check(&[], &[], &[]);
        check(&[0, 0], &[], &[]);
        check(&[0, 0], &[1], &[]);
        check(&[1, 0], &[1], &[1]);
        check(&[1, 0, 1], &[1], &[1, 0, 1]);
        check(&[1, 1], &[1, 1], &[1, 2, 1]);
        check(&[1, 1], &[1, 0, 1], &[1, 1, 1, 1]);
        check(&[0, 0, 1], &[0, 0, 1], &[0, 0, 0, 0, 1]);
    }

    #[test]
    fn eval() {
        fn check<F: Fn(i32) -> i32>(pol: &[i32], f: F) {
            for n in -10..10 {
                assert_eq!(f(n), Polynomial::new(pol.to_vec()).eval(n));
            }
        }
        check(&[], |_x| 0);
        check(&[1], |_x| 1);
        check(&[1, 1], |x| x + 1);
        check(&[0, 1], |x| x);
        check(&[10, -10, 10], |x| 10 * x * x - 10 * x + 10);
    }

    #[test]
    fn pretty() {
        fn check(slice: &[i32], s: &str) {
            assert_eq!(s.to_string(), Polynomial::new(slice.to_vec()).pretty("x"));
        }
        check(&[0], "0");
        check(&[1], "1");
        check(&[1, 1], "1+x");
        check(&[1, 1, 1], "1+x+x^2");
        check(&[2, 2, 2], "2+2*x+2*x^2");
        check(&[0, 0, 0, 1], "x^3");
        check(&[0, 0, 0, -1], "-x^3");
        check(&[-1, 0, 0, -1], "-1-x^3");
        check(&[-1, 1, 0, -1], "-1+x-x^3");
        check(&[-1, 1, -1, -1], "-1+x-x^2-x^3");
    }

    #[test]
    fn lagrange() {
        // Evaluate the lagrange polynomial at the x coordinates.
        // The error should be close to zero.
        fn check(xs: &[f64], ys: &[f64]) {
            let p = Polynomial::lagrange(xs, ys).unwrap();
            for (x, y) in xs.iter().zip(ys) {
                assert!((p.eval(*x) - y).abs() < 1e-9);
            }
        }

        // Squares
        check(&[1., 2., 3.], &[10., 40., 90.]);
        // Cubes
        check(&[-1., 0., 1., 2.], &[-1000., 0., 1000., 8000.]);
        // Non linear x.
        check(&[1., 9., 10., 11.], &[1., 2., 3., 4.]);
        // Test double x failure case.
        assert_eq!(
            Polynomial::lagrange(&[1., 9., 9., 11.], &[1., 2., 3., 4.]),
            None
        );
    }

    #[test]
    fn chebyshev() {
        // Construct a Chebyshev approximation for a function
        // and evaulate it at 100 points.
        fn check<F: Fn(f64) -> f64>(f: &F, n: usize, xmin: f64, xmax: f64) {
            let p = Polynomial::chebyshev(f, n, xmin, xmax).unwrap();
            for i in 0..=100 {
                let x = xmin + (i as f64) * ((xmax - xmin) / 100.0);
                let diff = (f(x) - p.eval(x)).abs();
                assert!(diff < 0.0001);
            }
        }

        // Approximate some common functions.
        use core::f64::consts::PI;
        check(&f64::sin, 7, -PI / 2., PI / 2.);
        check(&f64::cos, 7, 0., PI / 4.);

        // Test n >= 1 condition
        assert!(Polynomial::chebyshev(&f64::exp, 0, 0., 1.).is_none());

        // Test xmax > xmin condition
        assert!(Polynomial::chebyshev(&f64::ln, 1, 1., 0.).is_none());
    }
}

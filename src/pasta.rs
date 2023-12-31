// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(improper_ctypes)]
#![allow(unused)]

extern crate semolina;

use pasta_curves::pallas;

#[cfg(feature = "cuda")]
use crate::{cuda, cuda_available, CUDA_OFF};

extern "C" {
    fn mult_pippenger_pallas(
        out: *mut pallas::Point,
        points: *const pallas::Affine,
        npoints: usize,
        scalars: *const pallas::Scalar,
        is_mont: bool,
    );
}

pub fn pallas(
    points: &[pallas::Affine],
    scalars: &[pallas::Scalar],
) -> pallas::Point {
    let npoints = points.len();
    assert_eq!(npoints, scalars.len(), "length mismatch");

    #[cfg(feature = "cuda")]
    if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
        extern "C" {
            fn cuda_pippenger_pallas(
                out: *mut pallas::Point,
                points: *const pallas::Affine,
                npoints: usize,
                scalars: *const pallas::Scalar,
                is_mont: bool,
            ) -> cuda::Error;

        }
        let mut ret = pallas::Point::default();
        let err = unsafe {
            cuda_pippenger_pallas(
                &mut ret,
                &points[0],
                npoints,
                &scalars[0],
                true,
            )
        };
        assert!(err.code == 0, "{}", String::from(err));

        return ret;
    }
    let mut ret = pallas::Point::default();
    unsafe {
        mult_pippenger_pallas(&mut ret, &points[0], npoints, &scalars[0], true)
    };
    ret
}

use pasta_curves::vesta;

extern "C" {
    fn mult_pippenger_vesta(
        out: *mut vesta::Point,
        points: *const vesta::Affine,
        npoints: usize,
        scalars: *const vesta::Scalar,
        is_mont: bool,
    );
}

pub fn vesta(
    points: &[vesta::Affine],
    scalars: &[vesta::Scalar],
) -> vesta::Point {
    let npoints = points.len();
    assert_eq!(npoints, scalars.len(), "length mismatch");

    #[cfg(feature = "cuda")]
    if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
        extern "C" {
            fn cuda_pippenger_vesta(
                out: *mut vesta::Point,
                points: *const vesta::Affine,
                npoints: usize,
                scalars: *const vesta::Scalar,
                is_mont: bool,
            ) -> cuda::Error;

        }
        let mut ret = vesta::Point::default();
        let err = unsafe {
            cuda_pippenger_vesta(
                &mut ret,
                &points[0],
                npoints,
                &scalars[0],
                true,
            )
        };
        assert!(err.code == 0, "{}", String::from(err));

        return ret;
    }
    let mut ret = vesta::Point::default();
    unsafe {
        mult_pippenger_vesta(&mut ret, &points[0], npoints, &scalars[0], true)
    };
    ret
}

pub mod utils {
    use std::{
        mem::transmute,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use pasta_curves::{
        arithmetic::CurveExt,
        group::{ff::Field, Curve},
        pallas,
    };
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use rayon::iter::{
        IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator,
    };

    pub fn gen_points(npoints: usize) -> Vec<pallas::Affine> {
        let ret = vec![pallas::Affine::default(); npoints];

        let mut rnd = vec![0u8; 32 * npoints];
        ChaCha20Rng::from_entropy().fill_bytes(&mut rnd);

        let n_workers = rayon::current_num_threads();
        let work = AtomicUsize::new(0);
        rayon::scope(|s| {
            for _ in 0..n_workers {
                s.spawn(|_| {
                    let hash = pallas::Point::hash_to_curve("foobar");

                    let mut stride = 1024;
                    let mut tmp = vec![pallas::Point::default(); stride];

                    loop {
                        let work = work.fetch_add(stride, Ordering::Relaxed);
                        if work >= npoints {
                            break;
                        }
                        if work + stride > npoints {
                            stride = npoints - work;
                            unsafe { tmp.set_len(stride) };
                        }
                        for (i, point) in
                            tmp.iter_mut().enumerate().take(stride)
                        {
                            let off = (work + i) * 32;
                            *point = hash(&rnd[off..off + 32]);
                        }
                        #[allow(mutable_transmutes)]
                        pallas::Point::batch_normalize(&tmp, unsafe {
                            transmute::<
                                &[pallas::Affine],
                                &mut [pallas::Affine],
                            >(
                                &ret[work..work + stride]
                            )
                        });
                    }
                })
            }
        });

        ret
    }

    pub fn gen_scalars(npoints: usize) -> Vec<pallas::Scalar> {
        let ret =
            Arc::new(Mutex::new(vec![pallas::Scalar::default(); npoints]));

        let n_workers = rayon::current_num_threads();
        let work = Arc::new(AtomicUsize::new(0));

        rayon::scope(|s| {
            for _ in 0..n_workers {
                let ret_clone = Arc::clone(&ret);
                let work_clone = Arc::clone(&work);

                s.spawn(move |_| {
                    let mut rng = ChaCha20Rng::from_entropy();
                    loop {
                        let work = work_clone.fetch_add(1, Ordering::Relaxed);
                        if work >= npoints {
                            break;
                        }
                        let mut ret = ret_clone.lock().unwrap();
                        ret[work] = pallas::Scalar::random(&mut rng);
                    }
                });
            }
        });

        Arc::try_unwrap(ret).unwrap().into_inner().unwrap()
    }

    pub fn naive_multiscalar_mul(
        points: &[pallas::Affine],
        scalars: &[pallas::Scalar],
    ) -> pallas::Affine {
        let ret: pallas::Point = points
            .par_iter()
            .zip_eq(scalars.par_iter())
            .map(|(p, s)| p * s)
            .sum();

        ret.to_affine()
    }
}

#[cfg(test)]
mod tests {
    use pasta_curves::group::Curve;

    use crate::pasta::{
        pallas,
        utils::{gen_points, gen_scalars, naive_multiscalar_mul},
    };

    #[test]
    fn it_works() {
        #[cfg(not(debug_assertions))]
        const NPOINTS: usize = 128 * 1024;
        #[cfg(debug_assertions)]
        const NPOINTS: usize = 8 * 1024;

        let points = gen_points(NPOINTS);
        let scalars = gen_scalars(NPOINTS);

        let naive = naive_multiscalar_mul(&points, &scalars);
        println!("{:?}", naive);

        let ret = pallas(&points, &scalars).to_affine();
        println!("{:?}", ret);

        assert_eq!(ret, naive);
    }
}

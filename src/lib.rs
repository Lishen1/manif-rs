pub mod lie_group_base;
pub mod se2;
pub mod so3;
pub mod tangent_base;
pub use lie_group_base::LieGroupBase;
pub use tangent_base::TangentBase;

#[cfg(test)]
mod tests {
    use crate::{lie_group_base::LieGroupBase, tangent_base::TangentBase};
    use matrixcompare::assert_matrix_eq;
    use nalgebra::{UnitQuaternion, Vector3};
    use rand::Rng;
    ///////////////////////////////////
    fn get_w() -> Vector3<f64> {
        Vector3::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0) * 1e-2)
    }

    struct TestData {
        state: UnitQuaternion<f64>,
        state_other: UnitQuaternion<f64>,
        delta: Vector3<f64>,
        delta_other: Vector3<f64>,
    }
    impl TestData {
        const CASES: usize = 2;
        fn test_case(case: usize) -> TestData {
            match case {
                0 => TestData {
                    state: UnitQuaternion::identity(),
                    state_other: UnitQuaternion::identity(),
                    delta: Vector3::zeros(),
                    delta_other: Vector3::zeros(),
                },
                1 => TestData {
                    state: Vector3::<f64>::from_fn(|_, _| {
                        rand::thread_rng().gen_range(-1.0..1.0) * 1e-8
                    })
                    .exp_map(),
                    state_other: Vector3::<f64>::from_fn(|_, _| {
                        rand::thread_rng().gen_range(-1.0..1.0) * 1e-8
                    })
                    .exp_map(),
                    delta: Vector3::<f64>::from_fn(|_, _| {
                        rand::thread_rng().gen_range(-1.0..1.0) * 1e-8
                    }),
                    delta_other: Vector3::<f64>::from_fn(|_, _| {
                        rand::thread_rng().gen_range(-1.0..1.0) * 1e-8
                    }),
                },
                _ => panic!(),
            }
        }
    }
    const MANIF_TOL: f64 = 1e-5;
    const MANIF_NEAR_TOL: f64 = 1e-4;
    const MAT_NEAR_TOL: f64 = 1e-8;

    #[test]
    fn eval_inverse_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);
            let state = case.state;
            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out = state.manif_inverse_j(Some(&mut j_out_lhs));
            let state_pert = state.plus(w.clone()).inverse();
            let state_lin = state_out.rplus(j_out_lhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            )
        }
    }
    #[test]
    fn eval_lift_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);
            let state = case.state;
            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out = state.log_map_j(Some(&mut j_out_lhs));
            let state_pert = state.plus(w.clone()).log_map();
            let state_lin = state_out + j_out_lhs * w;
            assert_matrix_eq!(state_pert, state_lin, comp = abs, tol = MANIF_NEAR_TOL)
        }
    }
    #[test]
    fn eval_exp_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);
            let delta = case.delta;
            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out = delta.exp_map_j(Some(&mut j_out_lhs));
            let state_pert = (delta + w).exp_map();
            let state_lin = state_out.plus(j_out_lhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
        }
    }
    #[test]
    fn eval_compose_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let state_other = case.state_other;
            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out =
                state.compose_j(state_other, Some(&mut j_out_lhs), Some(&mut j_out_rhs));
            let state_pert = state.plus(w).compose(state_other);
            let state_lin = state_out.plus(j_out_lhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
            let state_pert = state.compose(state_other.plus(w));
            let state_lin = state_out.plus(j_out_rhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
        }
    }
    #[test]
    fn eval_between_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let state_other = case.state_other;

            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out =
                state.between_j(state_other, Some(&mut j_out_lhs), Some(&mut j_out_rhs));
            let state_pert = state.plus(w.clone()).between(state_other);
            let state_lin = state_out.plus(j_out_lhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
            let state_pert = state.between(state_other.plus(w));
            let state_lin = state_out.plus(j_out_rhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
        }
    }
    #[test]
    fn eval_rplus_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let delta = case.delta;

            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out = state.rplus_j(delta, Some(&mut j_out_lhs), Some(&mut j_out_rhs));
            let state_pert = state.plus(w.clone()).rplus(delta);
            let state_lin = state_out.rplus(j_out_lhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
            let state_pert = state.rplus(delta + w);
            let state_lin = state_out.rplus(j_out_rhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
        }
    }
    #[test]
    fn eval_lplus_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let delta = case.delta;

            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out = state.lplus_j(delta, Some(&mut j_out_lhs), Some(&mut j_out_rhs));
            let state_pert = state.plus(w.clone()).lplus(delta);
            let state_lin = state_out.rplus(j_out_lhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
            let state_pert = state.lplus(delta + w);
            let state_lin = state_out.rplus(j_out_rhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
        }
    }
    #[test]
    fn eval_plus_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let delta = case.delta;

            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out = state.plus_j(delta, Some(&mut j_out_lhs), Some(&mut j_out_rhs));
            let state_pert = state.plus(w.clone()).plus(delta);
            let state_lin = state_out.rplus(j_out_lhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
            let state_pert = state.plus(delta + w);
            let state_lin = state_out.rplus(j_out_rhs * w);
            assert_matrix_eq!(
                state_pert.coords,
                state_lin.coords,
                comp = abs,
                tol = MANIF_NEAR_TOL
            );
        }
    }
    #[test]
    fn eval_rminus_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let state_other = case.state_other;

            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out = state.rminus_j(state_other, Some(&mut j_out_lhs), Some(&mut j_out_rhs));
            let state_pert = state.plus(w.clone()).rminus(state_other);
            let state_lin = state_out.plus(j_out_lhs * w);
            assert_matrix_eq!(state_pert, state_lin, comp = abs, tol = MANIF_NEAR_TOL);
            let state_pert = state.rminus(state_other.plus(w));
            let state_lin = state_out.plus(j_out_rhs * w);
            assert_matrix_eq!(state_pert, state_lin, comp = abs, tol = MANIF_NEAR_TOL);
        }
    }
    #[test]
    fn eval_lminus_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let state_other = case.state_other;

            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out = state.lminus_j(state_other, Some(&mut j_out_lhs), Some(&mut j_out_rhs));

            let state_pert = state.plus(w.clone()).lminus(state_other);
            let state_lin = state_out.plus(j_out_lhs * w);
            assert_matrix_eq!(state_pert, state_lin, comp = abs, tol = MANIF_NEAR_TOL);

            let state_pert = state.lminus(state_other.plus(w));
            let state_lin = state_out.plus(j_out_rhs * w);
            assert_matrix_eq!(state_pert, state_lin, comp = abs, tol = MANIF_NEAR_TOL);
        }
    }
    #[test]
    fn eval_minus_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let state_other = case.state_other;

            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let w = get_w();
            let state_out = state.minus_j(state_other, Some(&mut j_out_lhs), Some(&mut j_out_rhs));
            let state_pert = state.plus(w.clone()).minus(state_other);
            let state_lin = state_out.plus(j_out_lhs * w);
            assert_matrix_eq!(state_pert, state_lin, comp = abs, tol = MANIF_NEAR_TOL);
            let state_pert = state.minus(state_other.plus(w));
            let state_lin = state_out.plus(j_out_rhs * w);
            assert_matrix_eq!(state_pert, state_lin, comp = abs, tol = MANIF_NEAR_TOL);
        }
    }
    #[test]
    fn eval_adj() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let state_other = case.state_other;
            let delta = case.delta;

            let adj_a = state.adj();
            let adj_b = state_other.adj();
            let adj_c = state.compose(state_other).adj();

            assert_matrix_eq!(adj_a * adj_b, adj_c, comp = abs, tol = 1e-8);
            assert_matrix_eq!(
                state.plus(delta).coords,
                state.plus(state.adj() * delta).coords,
                comp = abs,
                tol = MANIF_TOL
            );
            assert_matrix_eq!(
                state.adj().try_inverse().unwrap(),
                state.inverse().adj(),
                comp = abs,
                tol = MAT_NEAR_TOL
            );
        }
    }
    #[test]
    fn eval_adj_jl_jr() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let tan = state.log_map();
            let adj = state.adj();

            let jr = tan.rjac();
            let jl = tan.ljac();
            assert_matrix_eq!(jl, (-tan).rjac());
            assert_matrix_eq!(jl, adj.clone() * jr, comp = abs, tol = MAT_NEAR_TOL);
            assert_matrix_eq!(
                adj,
                jl * jr.try_inverse().unwrap(),
                comp = abs,
                tol = MAT_NEAR_TOL
            );
        }
    }
    #[test]
    fn eval_jr_jrinv_jl_jlinv() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);

            let state = case.state;
            let tan = state.log_map();
            let jr = tan.rjac();
            let jl = tan.ljac();
            let jrinv = tan.rjacinv();
            let jlinv = tan.ljacinv();

            let eye = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::identity();
            assert_matrix_eq!(eye, jr * jrinv, comp = abs, tol = MAT_NEAR_TOL);
            assert_matrix_eq!(eye, jl * jlinv, comp = abs, tol = MAT_NEAR_TOL);
        }
    }
    #[test]
    fn eval_act_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);
            let point = Vector3::<f64>::from_fn(|_i, _| rand::thread_rng().gen_range(-1.0..1.0));
            let mut j_pout_s = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_pout_p = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let state = case.state;
            let pointout = state.act_j(point, Some(&mut j_pout_s), Some(&mut j_pout_p));
            let w = get_w() * 1e-1;
            let w_order = 1e-4;
            let w_point =
                Vector3::<f64>::from_fn(|_i, _| rand::thread_rng().gen_range(-1.0..1.0) * w_order);

            let point_pert = state.plus(w).act(point);
            let point_lin = pointout + j_pout_s * w;
            let tol = 1e-6;
            assert_matrix_eq!(point_pert, point_lin, comp = abs, tol = tol);

            let point_pert = state.act(point + w_point);
            let point_lin = pointout + j_pout_p * w_point;
            assert_matrix_eq!(point_pert, point_lin, comp = abs, tol = MAT_NEAR_TOL);
        }
    }
    #[test]
    fn eval_tan_plus_tan_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);
            let delta = case.delta;
            let delta_other = case.delta_other;
            let w = get_w();
            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let delta_out = delta.plus_j(delta_other, Some(&mut j_out_lhs), Some(&mut j_out_rhs));

            let delta_pert = (delta + w).plus(delta_other);
            let delta_lin = delta_out.plus(j_out_lhs * w);
            assert_matrix_eq!(delta_pert, delta_lin, comp = abs, tol = MANIF_NEAR_TOL);

            let delta_pert = delta.plus(delta_other + w);
            let delta_lin = delta_out.plus(j_out_rhs * w);
            assert_matrix_eq!(delta_pert, delta_lin, comp = abs, tol = MANIF_NEAR_TOL);
        }
    }
    #[test]
    fn eval_tan_minus_tan_jac() {
        for i in 0..TestData::CASES {
            let case = TestData::test_case(i);
            let delta = case.delta;
            let delta_other = case.delta_other;
            let w = get_w();
            let mut j_out_lhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let mut j_out_rhs = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
            let delta_out = delta.minus_j(delta_other, Some(&mut j_out_lhs), Some(&mut j_out_rhs));

            let delta_pert = (delta + w).minus(delta_other);
            let delta_lin = delta_out.plus(j_out_lhs * w);
            assert_matrix_eq!(delta_pert, delta_lin, comp = abs, tol = MAT_NEAR_TOL);

            let delta_pert = delta.minus(delta_other + w);
            let delta_lin = delta_out.plus(j_out_rhs * w);
            assert_matrix_eq!(delta_pert, delta_lin, comp = abs, tol = MAT_NEAR_TOL);
        }
    }
}

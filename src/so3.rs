use std::ops::AddAssign;

use nalgebra::{
    convert, ArrayStorage, Matrix, Matrix3, Quaternion, RealField, Unit, UnitQuaternion,
};
use nalgebra::{Const, Vector3};

use crate::lie_group_base::ManifIdentity;
pub use crate::{lie_group_base::LieGroupBase, tangent_base::TangentBase};

impl<T: RealField> ManifIdentity<T> for Matrix<T, Const<3>, Const<3>, ArrayStorage<T, 3, 3>> {
    fn manif_identity() -> Self {
        Self::identity()
    }
}
#[allow(non_snake_case)]
impl<T: RealField> LieGroupBase for UnitQuaternion<T> {
    // NOTE: size related constants should not be a generic parameters
    // but no way do it now in rust
    const DIM: usize = 3;
    const DOF: usize = 3;
    const REP_SIZE: usize = 4;

    type T = T;
    type Jacobian = Matrix<T, Const<3>, Const<3>, ArrayStorage<T, 3, 3>>;
    type Tangent = Vector3<T>;
    type Point = Vector3<T>;

    fn log_map_j(&self, J_t_m: Option<&mut Self::Jacobian>) -> Self::Tangent {
        let sin_angle_squared = self.vector().norm_squared();
        let log_coeff: T = if sin_angle_squared > T::default_epsilon() {
            let sin_angle = sin_angle_squared.sqrt();
            let cos_angle = self.scalar();
            let two_angle: T = convert::<f64, T>(2.0f64)
                * (if cos_angle < T::zero() {
                    -sin_angle.clone().atan2(-cos_angle)
                } else {
                    sin_angle.clone().atan2(cos_angle)
                });
            two_angle / sin_angle
        } else {
            convert(2.0f64)
        };

        let tan = self.vector() * log_coeff;
        if let Some(J_t_m) = J_t_m {
            J_t_m.clone_from(&(Self::Jacobian::identity() + tan.hat() * convert::<f64, T>(0.5)));
            let theta2 = tan.norm_squared();
            if theta2 > T::default_epsilon() {
                let theta = theta2.clone().sqrt();
                J_t_m.add_assign(
                    tan.hat()
                        * tan.hat()
                        * (T::one() / theta2.clone()
                            - (T::one() + theta.clone().cos())
                                / (convert::<f64, T>(2.0) * theta.clone() * theta.sin())),
                );
            }
        }
        tan
    }

    fn compose_j(
        &self,
        m: Self,
        J_mc_ma: Option<&mut Self::Jacobian>,
        J_mc_mb: Option<&mut Self::Jacobian>,
    ) -> Self {
        if let Some(J_mc_ma) = J_mc_ma {
            J_mc_ma.clone_from(&m.clone().to_rotation_matrix().matrix().transpose());
        }

        if let Some(J_mc_mb) = J_mc_mb {
            J_mc_mb.clone_from(&Self::Jacobian::identity());
        }
        self * m
    }
    fn manif_inverse_j(&self, J_minv_m: Option<&mut Self::Jacobian>) -> Self {
        if let Some(J_minv_m) = J_minv_m {
            J_minv_m.clone_from(&(-self.clone().to_rotation_matrix().matrix()));
        }
        self.inverse()
    }

    fn adj(&self) -> Self::Jacobian {
        self.clone().to_rotation_matrix().matrix().clone()
    }

    fn act_j(
        &self,
        v: Self::Point,
        J_vout_m: Option<&mut Self::Jacobian>,
        J_vout_v: Option<&mut Self::Jacobian>,
    ) -> Self::Point {
        let R = self.clone().to_rotation_matrix();
        let R = R.matrix();
        if let Some(J_vout_m) = J_vout_m {
            J_vout_m.copy_from(&(-R * v.hat()));
        }
        if let Some(J_vout_v) = J_vout_v {
            J_vout_v.copy_from(&R);
        }
        R * v
    }
}

#[allow(non_snake_case)]
impl<T: RealField> TangentBase for Vector3<T> {
    type T = T;
    type LieGroup = UnitQuaternion<T>;
    type Jacobian = <UnitQuaternion<T> as LieGroupBase>::Jacobian;
    type LieAlg = Matrix3<T>;

    fn exp_map_j(&self, J_m_t: Option<&mut Self::Jacobian>) -> Self::LieGroup {
        let theta_sq = self.norm_squared();
        if theta_sq > T::default_epsilon() {
            let theta = theta_sq.clone().sqrt();
            if let Some(J_m_t) = J_m_t {
                let W = self.hat();
                J_m_t.copy_from(
                    &(Self::Jacobian::identity()
                        - W.clone() * (T::one() - theta.clone().cos()) / theta_sq.clone()
                        + W.clone() * W * (theta.clone() - theta.clone().sin())
                            / (theta_sq * theta.clone())),
                );
            }
            Self::LieGroup::from_axis_angle(&Unit::<Vector3<T>>::new_normalize(self.clone()), theta)
        } else {
            if let Some(J_m_t) = J_m_t {
                let W = self.hat();
                J_m_t.copy_from(&(Self::Jacobian::identity() - W * convert::<f64, T>(0.5)));
            }
            Self::LieGroup::from_quaternion(Quaternion::<T>::from_parts(
                T::one(),
                self / convert::<f64, T>(2.0),
            ))
        }
    }
    fn ljac(&self) -> Self::Jacobian {
        let theta_sq = self.norm_squared();
        let W = self.hat();
        if theta_sq <= T::default_epsilon() {
            Self::Jacobian::identity() + W * convert::<f64, T>(0.5)
        } else {
            let theta = theta_sq.clone().sqrt();
            Self::Jacobian::identity()
                + W.clone() * (T::one() - theta.clone().cos()) / theta_sq.clone()
                + W.clone() * W * (theta.clone() - theta.clone().sin()) / (theta_sq * theta)
        }
    }
    fn rjac(&self) -> Self::Jacobian {
        self.ljac().transpose()
    }
    fn hat(&self) -> Self::LieAlg {
        let x = self.x.clone();
        let y = self.y.clone();
        let z = self.z.clone();
        let o: T = convert(0.0);

        Self::LieAlg::new(
            o.clone(),
            -z.clone(),
            y.clone(),
            z.clone(),
            o.clone(),
            -x.clone(),
            -y.clone(),
            x.clone(),
            o.clone(),
        )
    }

    fn ljacinv(&self) -> Self::Jacobian {
        let theta_sq = self.norm_squared();
        let W = self.hat();
        if theta_sq <= T::default_epsilon() {
            Self::Jacobian::identity() - W * convert::<f64, T>(0.5)
        } else {
            let theta = theta_sq.clone().sqrt();
            Self::Jacobian::identity() - W.clone() * convert::<f64, T>(0.5)
                + W.clone()
                    * W
                    * (T::one() / theta_sq
                        - (T::one() + theta.clone().cos())
                            / (convert::<f64, T>(2.0) * theta.clone() * theta.sin()))
        }
    }

    fn rjacinv(&self) -> Self::Jacobian {
        self.ljacinv().transpose()
    }
}
#[cfg(test)]
mod tests {

    use super::*;
    use matrixcompare::assert_matrix_eq;
    use nalgebra::{Quaternion, UnitQuaternion, Vector3, Vector4};
    use rand::Rng;
    #[test]
    fn so3_tangent_skew() {
        let so3_lie = Vector3::<f64>::new(1.0, 2.0, 3.0);
        let w = so3_lie.hat();

        assert_eq!(0.0, w[(0, 0)]);
        assert_eq!(-3.0, w[(0, 1)]);
        assert_eq!(2.0, w[(0, 2)]);
        assert_eq!(3.0, w[(1, 0)]);
        assert_eq!(0.0, w[(1, 1)]);
        assert_eq!(-1.0, w[(1, 2)]);
        assert_eq!(-2.0, w[(2, 0)]);
        assert_eq!(1.0, w[(2, 1)]);
        assert_eq!(0.0, w[(2, 2)]);
    }
    #[test]
    fn so3_act() {
        let so3 = UnitQuaternion::identity();
        let transformed_point = so3.act(Vector3::new(1.0, 1.0, 1.0));

        assert_matrix_eq!(
            Vector3::new(1.0, 1.0, 1.0),
            transformed_point,
            comp = abs,
            tol = 1e-15
        );
        let so3 = UnitQuaternion::from_euler_angles(f64::pi(), f64::pi() / 2.0, f64::pi() / 4.0);
        let transformed_point = so3.act(Vector3::new(1.0, 1.0, 1.0));
        assert_matrix_eq!(
            Vector3::new(0.0, -1.414213562373, -1.0),
            transformed_point,
            comp = abs,
            tol = 1e-12
        );
        let so3 = UnitQuaternion::from_euler_angles(f64::pi() / 4.0, -f64::pi() / 2.0, -f64::pi());
        let transformed_point = so3.act(Vector3::new(1.0, 1.0, 1.0));
        assert_matrix_eq!(
            Vector3::new(1.414213562373, 0.0, 1.0),
            transformed_point,
            comp = abs,
            tol = 1e-12
        );
    }
    #[test]
    fn so3_exp() {
        let so3t = Vector3::<f64>::zeros();
        let so3 = so3t.exp_map();
        assert_eq!(so3.i, 0.0);
        assert_eq!(so3.j, 0.0);
        assert_eq!(so3.k, 0.0);
        assert_eq!(so3.w, 1.0);
        let so3t = Vector3::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let so3 = so3t.exp_map();
        let so3n = -so3t;
        let so3_inv = so3n.exp_map();
        assert_eq!(so3.i, -so3_inv.i);
        assert_eq!(so3.j, -so3_inv.j);
        assert_eq!(so3.k, -so3_inv.k);
        assert_eq!(so3.w, so3_inv.w);
    }
    #[test]
    fn so3_log() {
        let so3 = UnitQuaternion::identity();
        let so3_log = so3.log_map();
        assert_matrix_eq!(so3_log, Vector3::<f64>::zeros());
        let so3 = UnitQuaternion::from_quaternion(Quaternion::<f64>::from_vector(
            Vector4::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ));
        let so3_log = so3.log_map();
        let so3_inv_log = so3.inverse().log_map();
        assert_matrix_eq!(so3_inv_log, -so3_log);
    }
    #[test]
    fn so3_inv_jac() {
        let so3 = UnitQuaternion::<f64>::identity();
        let mut j_inv = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
        let so3_inv = so3.manif_inverse_j(Some(&mut j_inv));
        assert_eq!(so3_inv.i, 0.0);
        assert_eq!(so3_inv.j, 0.0);
        assert_eq!(so3_inv.k, 0.0);
        assert_eq!(so3_inv.w, 1.0);
        assert_eq!(j_inv.nrows(), 3);
        assert_eq!(j_inv.ncols(), 3);
        assert_eq!(-1.0, j_inv[(0, 0)]);
        assert_eq!(0.0, j_inv[(0, 1)]);
        assert_eq!(0.0, j_inv[(0, 2)]);
        assert_eq!(0.0, j_inv[(1, 0)]);
        assert_eq!(-1.0, j_inv[(1, 1)]);
        assert_eq!(0.0, j_inv[(1, 2)]);
        assert_eq!(0.0, j_inv[(2, 0)]);
        assert_eq!(0.0, j_inv[(2, 1)]);
        assert_eq!(-1.0, j_inv[(2, 2)]);

        let so3 = UnitQuaternion::from_quaternion(Quaternion::<f64>::from_vector(
            Vector4::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ));
        let so3_inv = so3.manif_inverse_j(Some(&mut j_inv));
        assert_eq!(-so3.i, so3_inv.i);
        assert_eq!(-so3.j, so3_inv.j);
        assert_eq!(-so3.k, so3_inv.k);
        assert_eq!(so3.w, so3_inv.w);

        assert_eq!(3, j_inv.nrows());
        assert_eq!(3, j_inv.ncols());

        let rot = so3.clone().to_rotation_matrix().matrix().clone();

        assert_eq!(-rot[(0, 0)], j_inv[(0, 0)]);
        assert_eq!(-rot[(0, 1)], j_inv[(0, 1)]);
        assert_eq!(-rot[(0, 2)], j_inv[(0, 2)]);
        assert_eq!(-rot[(1, 0)], j_inv[(1, 0)]);
        assert_eq!(-rot[(1, 1)], j_inv[(1, 1)]);
        assert_eq!(-rot[(1, 2)], j_inv[(1, 2)]);
        assert_eq!(-rot[(2, 0)], j_inv[(2, 0)]);
        assert_eq!(-rot[(2, 1)], j_inv[(2, 1)]);
        assert_eq!(-rot[(2, 2)], j_inv[(2, 2)]);
    }
    #[test]
    fn so3_left_jac() {
        let so3 = UnitQuaternion::<f64>::identity();
        let mut j_inv = <UnitQuaternion<f64> as LieGroupBase>::Jacobian::zeros();
        let so3_log = so3.log_map_j(Some(&mut j_inv));
        assert_eq!(so3_log.x, 0.0);
        assert_eq!(so3_log.y, 0.0);
        assert_eq!(so3_log.z, 0.0);

        assert_eq!(3, j_inv.nrows());
        assert_eq!(3, j_inv.ncols());
        //TODO: add jac test
    }
    #[test]
    fn so3_right_left_jac_adj() {
        let tan = Vector3::<f64>::zeros();
        assert_matrix_eq!(
            tan.ljac(),
            tan.exp_map().clone().to_rotation_matrix().matrix() * tan.rjac(),
            comp = abs,
            tol = 1e-9
        );

        let tan = Vector3::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        assert_matrix_eq!(
            tan.ljac(),
            tan.exp_map().clone().to_rotation_matrix().matrix() * tan.rjac(),
            comp = abs,
            tol = 1e-9
        );
    }
    #[test]
    fn so3_right_left_jac() {
        let tan = Vector3::<f64>::zeros();
        assert_matrix_eq!(tan.ljac(), tan.rjac().transpose());
        assert_matrix_eq!(tan.rjac(), tan.ljac().transpose());

        let tan = Vector3::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        assert_matrix_eq!(tan.ljac(), tan.rjac().transpose());
        assert_matrix_eq!(tan.rjac(), tan.ljac().transpose());
    }
    #[test]
    fn so3_rplus() {
        let so3a = UnitQuaternion::<f64>::identity();
        let so3b = Vector3::<f64>::zeros();
        let so3c = so3a.rplus(so3b);
        assert_eq!(0.0, so3c.i);
        assert_eq!(0.0, so3c.j);
        assert_eq!(0.0, so3c.k);
        assert_eq!(1.0, so3c.w);
        let so3a = UnitQuaternion::from_quaternion(Quaternion::<f64>::from_vector(
            Vector4::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ));
        let so3c = so3a.rplus(so3b);
        assert_eq!(so3a.i, so3c.i);
        assert_eq!(so3a.j, so3c.j);
        assert_eq!(so3a.k, so3c.k);
        assert_eq!(so3a.w, so3c.w);
    }
    #[test]
    fn so3_lplus() {
        let so3a = UnitQuaternion::<f64>::identity();
        let so3b = Vector3::<f64>::zeros();
        let so3c = so3a.lplus(so3b);
        assert_eq!(0.0, so3c.i);
        assert_eq!(0.0, so3c.j);
        assert_eq!(0.0, so3c.k);
        assert_eq!(1.0, so3c.w);
        let so3a = UnitQuaternion::from_quaternion(Quaternion::<f64>::from_vector(
            Vector4::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ));
        let so3c = so3a.lplus(so3b);
        assert_eq!(so3a.i, so3c.i);
        assert_eq!(so3a.j, so3c.j);
        assert_eq!(so3a.k, so3c.k);
        assert_eq!(so3a.w, so3c.w);
    }
    #[test]
    fn so3_plus() {
        let so3a = UnitQuaternion::from_quaternion(Quaternion::<f64>::from_vector(
            Vector4::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ));
        let so3t = Vector3::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0));
        let so3c = so3a.plus(so3t);
        let so3d = so3a.rplus(so3t);
        assert_eq!(so3c.i, so3d.i);
        assert_eq!(so3c.j, so3d.j);
        assert_eq!(so3c.k, so3d.k);
        assert_eq!(so3c.w, so3d.w);
    }
    #[test]
    fn so3_rminus() {
        let so3a = UnitQuaternion::<f64>::identity();
        let so3b = UnitQuaternion::<f64>::identity();
        let so3c = so3a.rminus(so3b);
        assert_matrix_eq!(so3c, Vector3::<f64>::zeros());
        let so3a = UnitQuaternion::from_quaternion(Quaternion::<f64>::from_vector(
            Vector4::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ));
        let so3b = so3a.clone();
        let so3c = so3a.rminus(so3b);
        assert_matrix_eq!(so3c, Vector3::<f64>::zeros(), comp = abs, tol = 1e-15);
    }
    #[test]
    fn so3_lminus() {
        let so3a = UnitQuaternion::<f64>::identity();
        let so3b = UnitQuaternion::<f64>::identity();
        let so3c = so3a.lminus(so3b);
        assert_matrix_eq!(so3c, Vector3::<f64>::zeros());
        let so3a = UnitQuaternion::from_quaternion(Quaternion::<f64>::from_vector(
            Vector4::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ));
        let so3b = so3a.clone();
        let so3c = so3a.lminus(so3b);
        assert_matrix_eq!(so3c, Vector3::<f64>::zeros(), comp = abs, tol = 1e-15);
    }
    #[test]
    fn so3_minus() {
        let so3a = UnitQuaternion::from_quaternion(Quaternion::<f64>::from_vector(
            Vector4::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ));
        let so3b = UnitQuaternion::from_quaternion(Quaternion::<f64>::from_vector(
            Vector4::<f64>::from_fn(|_, _| rand::thread_rng().gen_range(-1.0..1.0)),
        ));
        let so3c = so3a.minus(so3b);
        let so3d = so3a.rminus(so3b);

        assert_matrix_eq!(so3c, so3d);
    }
}

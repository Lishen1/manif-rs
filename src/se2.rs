use core::ops::Add;
use core::ops::Neg;
use core::ops::Sub;

use crate::{LieGroupBase, TangentBase};
use nalgebra::convert;
use nalgebra::Complex;
use nalgebra::Translation2;
use nalgebra::UnitComplex;
use nalgebra::U2;
use nalgebra::{Isometry2, Matrix3, RealField, Vector2, Vector3};
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct SE2Tangent<T: RealField>(pub Vector3<T>);
impl<T: RealField + Copy> TangentBase for SE2Tangent<T> {
    type T = T;
    type LieGroup = Isometry2<T>;
    type Jacobian = <Isometry2<T> as LieGroupBase>::Jacobian;
    type LieAlg = Matrix3<T>;

    fn hat(&self) -> Self::LieAlg {
        todo!()
    }

    #[allow(non_snake_case)]
    fn exp_map_j(&self, J_m_t: Option<&mut Self::Jacobian>) -> Self::LieGroup {
        let theta = self.angle();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let theta_sq = theta * theta;

        let A: T; // sin_theta_by_theta
        let B: T; // one_minus_cos_theta_by_theta

        if theta_sq < T::default_epsilon() {
            // Taylor approximation
            A = convert::<_, T>(1.0) - convert::<_, T>(1. / 6.) * theta_sq;
            B = convert::<_, T>(0.5) * theta - convert::<_, T>(1. / 24.) * theta * theta_sq;
        } else {
            // Euler
            A = sin_theta / theta;
            B = (convert::<_, T>(1.0) - cos_theta) / theta;
        }

        if let Some(J_m_t) = J_m_t {
            // Jr
            J_m_t.fill_with_identity();
            J_m_t[(0, 0)] = A;
            J_m_t[(0, 1)] = B;
            J_m_t[(1, 0)] = -B;
            J_m_t[(1, 1)] = A;

            if theta_sq < T::default_epsilon() {
                J_m_t[(0, 2)] = -self.y() / convert(2.0) + theta * self.x() / convert(6.0);
                J_m_t[(1, 2)] = self.x() / convert(2.0) + theta * self.y() / convert(6.0);
            } else {
                J_m_t[(0, 2)] = (-self.y() + theta * self.x() + self.y() * cos_theta
                    - self.x() * sin_theta)
                    / theta_sq;
                J_m_t[(1, 2)] =
                    (self.x() + theta * self.y() - self.x() * cos_theta - self.y() * sin_theta)
                        / theta_sq;
            }
        }
        Self::LieGroup::from_parts(
            Translation2::new(A * self.x() - B * self.y(), B * self.x() + A * self.y()),
            UnitComplex::new_normalize(Complex::new(cos_theta, sin_theta)),
        )
    }

    fn rjac(&self) -> Self::Jacobian {
        todo!()
    }

    fn ljac(&self) -> Self::Jacobian {
        todo!()
    }

    fn rjacinv(&self) -> Self::Jacobian {
        todo!()
    }

    fn ljacinv(&self) -> Self::Jacobian {
        todo!()
    }
}
impl<T: RealField> Add for SE2Tangent<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
impl<T: RealField> Sub for SE2Tangent<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
impl<T: RealField> Neg for SE2Tangent<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        todo!()
    }
}
impl<T: RealField + Copy> SE2Tangent<T> {
    pub fn angle(&self) -> T {
        self.0[2]
    }
    pub fn x(&self) -> T {
        self.0[0]
    }
    pub fn y(&self) -> T {
        self.0[1]
    }
}
#[allow(non_snake_case)]
impl<T: RealField + Copy> LieGroupBase for Isometry2<T> {
    const DIM: usize = 2;

    const DOF: usize = 3;

    const REP_SIZE: usize = 4;

    type T = T;

    type Jacobian = Matrix3<T>;

    type Tangent = SE2Tangent<T>;

    type Point = Vector2<T>;

    fn manif_inverse_j(&self, J_minv_m: Option<&mut Self::Jacobian>) -> Self {
        todo!()
    }

    fn log_map_j(&self, J_t_m: Option<&mut Self::Jacobian>) -> Self::Tangent {
        let theta = self.rotation.angle();
        let cos_theta = self.rotation.re;
        let sin_theta = self.rotation.im;
        let theta_sq = theta * theta;

        let A: T; // sin_theta_by_theta
        let B: T; // one_minus_cos_theta_by_theta

        if theta_sq < T::default_epsilon() {
            // Taylor approximation
            A = convert::<_, T>(1.0) - convert::<_, T>(1. / 6.) * theta_sq;
            B = convert::<_, T>(0.5) * theta - convert::<_, T>(1. / 24.) * theta * theta_sq;
        } else {
            // Euler
            A = sin_theta / theta;
            B = (convert::<_, T>(1.0) - cos_theta) / theta;
        }

        let den = convert::<_, T>(1.0) / (A * A + B * B);

        let A = A * den;
        let B = B * den;

        let x = self.translation.x;
        let y = self.translation.y;
        let tan = SE2Tangent(Vector3::new(A * x + B * y, -B * x + A * y, theta));

        if let Some(J_t_m) = J_t_m {
            // Jr^-1
            J_t_m.clone_from(&tan.rjacinv());
        }

        tan
    }

    fn compose_j(
        &self,
        m: Self,
        J_mc_ma: Option<&mut Self::Jacobian>,
        J_mc_mb: Option<&mut Self::Jacobian>,
    ) -> Self {
        todo!()
    }

    fn act_j(
        &self,
        v: Self::Point,
        J_vout_m: Option<&mut Self::Jacobian>,
        J_vout_v: Option<&mut Self::Jacobian>,
    ) -> Self::Point {
        todo!()
    }

    fn adj(&self) -> Self::Jacobian {
        let mut Adj = Self::Jacobian::identity();
        Adj.generic_view_mut((0, 0), (U2, U2))
            .copy_from(&self.rotation.to_rotation_matrix().matrix());
        let x = self.translation.x;
        let y = self.translation.y;
        Adj[(0, 2)] = y;
        Adj[(1, 2)] = -x;
        return Adj;
    }
}
#[cfg(test)]
mod tests {
    use crate::{LieGroupBase, TangentBase};
    use matrixcompare::assert_matrix_eq;
    use nalgebra::{Isometry2, Vector2, Vector3};

    use super::SE2Tangent;
    #[test]
    fn exp() {
        let mut se2 = Isometry2::<f64>::new(Vector2::zeros(), 0.0);
        let lm = se2.log_map();
        let t = SE2Tangent(Vector3::new(1.0, -1.0, std::f64::consts::PI));
        let m = t.exp_map();
        assert_matrix_eq!(m.log_map().0, t.0, comp = abs, tol = 1e-15);
    }
}

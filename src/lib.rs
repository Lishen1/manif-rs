use nalgebra::{ArrayStorage, DMatrixViewMut, Matrix, Matrix3x6, RealField, UnitQuaternion};
use nalgebra::{Const, Vector3};
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[allow(non_snake_case)]
pub trait LieGroupBase {
    const DIM: usize;
    const DOF: usize;
    const REP_SIZE: usize;
    type T: RealField;
    // type LieGroup: LieGroup<R>;
    // type Jacobian =
    //     Matrix<R, Const<Self::DOF>, Const<Self::DOF>, ArrayStorage<R, Self::DOF, Self::DOF>>;
    // type OptJacobianRef = Option<DMatrixViewMut<>>

    type Jacobian;
    type Tangent: TangentBase;
    // type Tangent;
    fn log_map_j(&self, J_t_m: Option<&mut Self::Jacobian>) -> Self::Tangent;
    fn log_map(&self) -> Self::Tangent {
        self.log_map_j(None)
    }
    // fn rplus_j<TN: TangentBase>(
    //     &self,
    //     t: TN,
    //     J_mout_m: Option<&mut Self::Jacobian>,
    //     J_mout_t: Option<&mut Self::Jacobian>,
    // ) -> TN::Derived {
    //     t.exp_map()
    // }

    fn compose(
        &self,
        m: Self,
        J_mc_ma: Option<&mut Self::Jacobian>,
        J_mc_mb: Option<&mut Self::Jacobian>,
    ) -> impl LieGroupBase;
    fn rplus_j(
        &self,
        t: Self::Tangent,
        J_mout_m: Option<&mut Self::Jacobian>,
        J_mout_t: Option<&mut Self::Jacobian>,
    ) -> Self
    where
        Self: Sized,
    {
        if let Some(J_mount_t) = J_mout_t {
            todo!();
            // J_mount_t = t.rjac();
        }
        self.compose(t.exp_map(), J_mout_m, None)
        // self.compose(*self, J_mout_m, None)
        // t.exp_map().compose(self, None, None)
    }
}

#[allow(non_snake_case)]
pub trait TangentBase {
    type T: RealField;
    type Derived: LieGroupBase;
    type Jacobian;
    fn exp_map_j(&self, J_m_t: Option<&mut Self::Jacobian>) -> Self::Derived;
    fn exp_map(&self) -> Self::Derived {
        self.exp_map_j(None)
    }
}

///////////// impl /////////////////////////////////////

#[allow(non_snake_case)]
impl<T: RealField> LieGroupBase for UnitQuaternion<T> {
    // NOTE: size related constants should not be a generic parameters
    // but no way do it now in rust
    const DIM: usize = 3;
    const DOF: usize = 3;
    const REP_SIZE: usize = 4;

    type T = T;
    // type Jacobian = Matrix<
    //     T,
    //     Const<{ Self::DOF }>,
    //     Const<{ Self::DOF }>,
    //     ArrayStorage<T, { Self::DOF }, { Self::DOF }>,
    // >;
    type Jacobian = Matrix<T, Const<3>, Const<3>, ArrayStorage<T, 3, 3>>;
    type Tangent = Vector3<T>;

    fn log_map_j(&self, J_t_m: Option<&mut Self::Jacobian>) -> Self::Tangent {
        todo!()
    }

    fn compose(
        &self,
        m: Self,
        J_mc_ma: Option<&mut Self::Jacobian>,
        J_mc_mb: Option<&mut Self::Jacobian>,
    ) -> Self {
        self * m
    }
}

#[allow(non_snake_case)]
impl<T: RealField> TangentBase for Vector3<T> {
    type T = T;
    type Derived = UnitQuaternion<T>;
    type Jacobian = <UnitQuaternion<T> as LieGroupBase>::Jacobian;

    fn exp_map_j(&self, J_m_t: Option<&mut Self::Jacobian>) -> Self::Derived {
        let j = Self::Jacobian::identity();
        Self::Derived::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
        // let tangent = Vector3::<f64>::new(0.1, 0.3, 0.1);
        // let so3 = tangent.exp_map_j();
    }
}

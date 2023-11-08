use crate::{tangent_base::TangentBase, MakeIdentity};
use nalgebra::RealField;
use std::ops::{Mul, Neg};

#[allow(non_snake_case)]
/// Base class for Lie groups.
///
/// Defines the minimum common API for Lie groups.
///
/// See also the [TangentBase] for more details.
pub trait LieGroupBase:
    Sized + Clone + From<<<Self as LieGroupBase>::Tangent as TangentBase>::LieGroup>
{
    const DIM: usize;
    const DOF: usize;
    const REP_SIZE: usize;

    type T: RealField;
    type Jacobian: Clone
        + MakeIdentity<Self::T>
        + From<<Self::Tangent as TangentBase>::Jacobian>
        + Mul
        + Neg
        + From<<Self::Jacobian as Neg>::Output>
        + From<<Self::Jacobian as Mul>::Output>;
    type Tangent: TangentBase + Neg + Clone + From<<Self::Tangent as Neg>::Output>;
    /// Computes the inverse of the Lie group object `self`.
    ///
    /// **Optional Argument**
    ///
    /// * `J_m_t`: A parameter to store the Jacobian of the inverse with respect to `self`.
    ///
    /// **Returns**
    ///
    /// The inverse of `self`.
    ///
    /// **Note**
    ///
    /// This function implements Eq. (3).
    ///
    /// See also: [Eq. (3)](link-to-equation-3), [TangentBase](link-to-tangentbase).
    fn inverse_j(&self, J_minv_m: Option<&mut Self::Jacobian>) -> Self;
    /// Computes the corresponding Lie algebra element in vector form.
    ///
    /// Optionally takes a `J_t_m` parameter to store the Jacobian of the tangent with respect to this
    /// object.
    ///
    /// Returns the tangent element in vector form.
    ///
    /// **Note**
    ///
    /// This is the log() map in vector form.
    ///
    /// See also [Eq. (24)](link-to-equation-24).
    fn log_map_j(&self, J_t_m: Option<&mut Self::Jacobian>) -> Self::Tangent;
    fn log_map(&self) -> Self::Tangent {
        self.log_map_j(None)
    }
    /// Composes `self` with another element of the same Lie group.
    ///
    /// **Arguments**
    ///
    /// * `m`: Another element of the same Lie group.
    ///
    /// **Optional Arguments**
    ///
    /// * `J_mc_ma`: A parameter to store the Jacobian of the composition with respect to `self`.
    /// * `J_mc_mb`: A parameter to store the Jacobian of the composition with respect to `m`.
    ///
    /// **Returns**
    ///
    /// The composition of `self * m`.
    ///
    /// **Note**
    ///
    /// This function implements Eqs. (1,2,3,4).
    ///
    /// See also: [Eq. (1)](link-to-equation-1), [Eq. (2)](link-to-equation-2), [Eq. (3)](link-to-equation-3), [Eq. (4)](link-to-equation-4).
    fn compose_j(
        &self,
        m: Self,
        J_mc_ma: Option<&mut Self::Jacobian>,
        J_mc_mb: Option<&mut Self::Jacobian>,
    ) -> Self;
    fn compose(&self, m: Self) -> Self {
        self.compose_j(m, None, None)
    }
    /// Computes the Adjoint of the Lie group element `self`.
    ///
    /// **Note**
    ///
    /// This function implements Eq. (29).
    ///
    /// See also: [Eq. (29)](link-to-equation-29).
    fn adj(&self) -> Self::Jacobian;
    /// Computes the right oplus operation of the Lie group.
    ///
    /// **Arguments**
    ///
    /// * `t`: An element of the tangent of the Lie group.
    ///
    /// **Optional Arguments**
    ///
    /// * `J_mout_m`: A parameter to store the Jacobian of the oplus operation with respect to `self`.
    /// * `J_mout_t`: A parameter to store the Jacobian of the oplus operation with respect to the tangent element.
    ///
    /// **Returns**
    ///
    /// An element of the Lie group.
    ///
    /// **Note**
    ///
    /// This function implements Eq. (25).
    ///
    /// See also: [Eq. (25)](link-to-equation-25).
    fn rplus_j(
        &self,
        t: Self::Tangent,
        J_mout_m: Option<&mut Self::Jacobian>,
        J_mout_t: Option<&mut Self::Jacobian>,
    ) -> Self {
        if let Some(J_mount_t) = J_mout_t {
            J_mount_t.clone_from(&Self::Jacobian::from(t.rjac()));
        }
        self.compose_j(t.exp_map().into(), J_mout_m, None)
    }
    fn rplus(&self, t: Self::Tangent) -> Self {
        self.rplus_j(t, None, None)
    }
    /// Computes the left oplus operation of the Lie group.
    ///
    /// **Arguments**
    ///
    /// * `t`: An element of the tangent of the Lie group.
    ///
    /// **Optional Arguments**
    ///
    /// * `J_mout_m`: A parameter to store the Jacobian of the oplus operation with respect to `self`.
    /// * `J_mout_t`: A parameter to store the Jacobian of the oplus operation with respect to the tangent element.
    ///
    /// **Returns**
    ///
    /// An element of the Lie group.
    ///
    /// **Note**
    ///
    /// This function implements Eq. (27).
    ///
    /// See also: [Eq. (27)](link-to-equation-27).
    fn lplus_j(
        &self,
        t: Self::Tangent,
        J_mout_m: Option<&mut Self::Jacobian>,
        J_mout_t: Option<&mut Self::Jacobian>,
    ) -> Self {
        if let Some(J_mout_t) = J_mout_t {
            J_mout_t.clone_from(
                &((self.inverse_j(None).adj() * Self::Jacobian::from(t.rjac())).into()),
            );
        }
        if let Some(J_mout_m) = J_mout_m {
            J_mout_m.clone_from(&(Self::Jacobian::make_identity()));
        }
        Self::from(t.exp_map()).compose(self.clone())
    }
    fn lplus(&self, t: Self::Tangent) -> Self {
        self.lplus_j(t, None, None)
    }
    /// An alias for the right oplus operation.
    ///
    /// See also: [rplus](link-to-rplus).
    ///
    fn plus_j(
        &self,
        t: Self::Tangent,
        J_mout_m: Option<&mut Self::Jacobian>,
        J_mout_t: Option<&mut Self::Jacobian>,
    ) -> Self {
        self.rplus_j(t, J_mout_m, J_mout_t)
    }
    fn plus(&self, t: Self::Tangent) -> Self {
        self.plus_j(t, None, None)
    }
    /// Computes the right ominus operation of the Lie group.
    ///
    /// **Arguments**
    ///
    /// * `m`: Another element of the same Lie group.
    ///
    /// **Optional Arguments**
    ///
    /// * `J_t_ma`: A parameter to store the Jacobian of the ominus operation with respect to `self`.
    /// * `J_t_mb`: A parameter to store the Jacobian of the ominus operation with respect to the other element.
    ///
    /// **Returns**
    ///
    /// An element of the tangent space of the Lie group.
    ///
    /// **Note**
    ///
    /// This function implements Eq. (26).
    ///
    /// See also: [Eq. (26)](link-to-equation-26).
    fn rminus_j(
        &self,
        m: Self,
        J_t_ma: Option<&mut Self::Jacobian>,
        J_t_mb: Option<&mut Self::Jacobian>,
    ) -> Self::Tangent {
        let t = m.inverse_j(None).compose(self.clone()).log_map();
        if let Some(J_t_ma) = J_t_ma {
            J_t_ma.clone_from(&(Self::Jacobian::from(t.rjacinv())));
        }
        if let Some(J_t_mb) = J_t_mb {
            J_t_mb.clone_from(
                &(Self::Jacobian::from(-Self::Jacobian::from(
                    Self::Tangent::from(-t.clone()).rjacinv(),
                ))),
            );
        }
        t
    }
    fn rminus(&self, m: Self) -> Self::Tangent {
        self.rminus_j(m, None, None)
    }
    /// Computes the left ominus operation of the Lie group.
    ///
    /// **Arguments**
    ///
    /// * `m`: Another element of the same Lie group.
    ///
    /// **Optional Arguments**
    ///
    /// * `J_t_ma`: A parameter to store the Jacobian of the ominus operation with respect to `self`.
    /// * `J_t_mb`: A parameter to store the Jacobian of the ominus operation with respect to the other element.
    ///
    /// **Returns**
    ///
    /// An element of the tangent space of the Lie group.
    ///
    /// **Note**
    ///
    /// This function implements Eq. (28).
    ///
    /// See also: [Eq. (28)](link-to-equation-28).
    fn lminus_j(
        &self,
        m: Self,
        J_t_ma: Option<&mut Self::Jacobian>,
        J_t_mb: Option<&mut Self::Jacobian>,
    ) -> Self::Tangent {
        let t = self.compose(m.inverse_j(None)).log_map();
        if let Some(J_t_ma) = J_t_ma {
            J_t_ma.clone_from(&(Self::Jacobian::from(Self::Jacobian::from(t.rjacinv()) * m.adj())));
        }
        if let Some(J_t_mb) = J_t_mb {
            J_t_mb.clone_from(
                &(Self::Jacobian::from(-Self::Jacobian::from(
                    Self::Jacobian::from(t.rjacinv()) * m.adj(),
                ))),
            );
        }
        t
    }
    fn lminus(&self, m: Self) -> Self::Tangent {
        self.lminus_j(m, None, None)
    }
    /// An alias for the right ominus operation.

    /// See also: [rminus](link-to-rminus).
    fn minus_j(
        &self,
        m: Self,
        J_t_ma: Option<&mut Self::Jacobian>,
        J_t_mb: Option<&mut Self::Jacobian>,
    ) -> Self::Tangent {
        self.rminus_j(m, J_t_ma, J_t_mb)
    }
    fn minus(&self, m: Self) -> Self::Tangent {
        self.rminus_j(m, None, None)
    }
    /// Computes the composition of `self` with another element of the same Lie group.
    ///
    /// **Arguments**
    ///
    /// * `m`: Another element of the same Lie group. [Image of Lie group element]
    ///
    /// **Optional Arguments**
    ///
    /// * `J_mc_ma`: A parameter to store the Jacobian of the composition with respect to `self`. [Image of Jacobian]
    /// * `J_mc_mb`: A parameter to store the Jacobian of the composition with respect to `m`. [Image of Jacobian]
    ///
    /// **Returns**
    ///
    /// The composition of `self * m`. [Image of composition operation]
    ///
    /// **Note**
    ///
    /// This function implements Eqs. (1,2,3,4).
    ///
    /// See also: [Eq. (1)](link-to-equation-1), [Eq. (2)](link-to-equation-2), [Eq. (3)](link-to-equation-3), [Eq. (4)](link-to-equation-4).
    fn between_j(
        &self,
        m: Self,
        J_mc_ma: Option<&mut Self::Jacobian>,
        J_mc_mb: Option<&mut Self::Jacobian>,
    ) -> Self;
    fn between(&self, m: Self) -> Self {
        self.between_j(m, None, None)
    }
}

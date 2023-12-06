use crate::lie_group_base::{LieGroupBase, ManifIdentity};
use nalgebra::RealField;
use std::ops::{Add, Neg, Sub};

#[allow(non_snake_case)]
/// Base class for Lie groups' tangents.
///
/// Defines the minimum common API.
///
/// See also: [LieGroupBase].
pub trait TangentBase:
    Sized + Neg + Clone + Add<Self, Output = Self> + Sub<Self, Output = Self>
{
    type T: RealField;
    type LieGroup: LieGroupBase;
    type Jacobian: ManifIdentity<Self::T> + Clone + Neg + From<<Self::Jacobian as Neg>::Output>;
    type LieAlg;
    /// Computes the hat operator of the tangent element `self`.
    ///
    /// **Returns**
    ///
    /// The isomorphic element in the Lie algebra. [Image of Lie algebra element]
    ///
    /// **Note**
    ///
    /// This function implements Eq. (10).
    ///
    /// See also: [Eq. (10)](link-to-equation-10).
    fn hat(&self) -> Self::LieAlg;
    /// Computes the associated Lie group element for the tangent element `self`.
    ///
    /// **Optional Argument**
    ///
    /// * `J_m_t`: A parameter to store the Jacobian of the Lie group element with respect to `self`.
    ///
    /// **Returns**
    ///
    /// The associated Lie group element. [Image of Lie group element]
    ///
    /// **Note**
    ///
    /// This function implements the exponential map (exp()) with the argument in vector form.
    ///
    /// See also: [Eq. (23)](link-to-equation-23).
    fn exp_map_j(&self, J_m_t: Option<&mut Self::Jacobian>) -> Self::LieGroup;
    fn exp_map(&self) -> Self::LieGroup {
        self.exp_map_j(None)
    }
    /// Computes the right Jacobian of the Lie group element `self`.
    ///
    /// **Note**
    ///
    /// This is the right Jacobian of the exponential map (`exp`), often referred to as "the right Jacobian".
    ///
    /// See also:
    ///
    /// * [Eq. (41)](link-to-equation-41) for the right Jacobian of general functions.
    /// * [Eqs. (126, 143, 163, 179, 191)](link-to-equations-126-143-163-179-191) for implementations of the right Jacobian of `exp`.
    fn rjac(&self) -> Self::Jacobian;
    /// Computes the left Jacobian of the Lie group element `self`.
    ///
    /// **Note**
    ///
    /// This is the left Jacobian of the exponential map (`exp`), often referred to as "the left Jacobian".
    ///
    /// See also:
    ///
    /// * [Eq. (44)](link-to-equation-44) for the left Jacobian of general functions.
    /// * [Eqs. (126, 145, 164, 179, 191)](link-to-equations-126-145-164-179-191) for implementations of the left Jacobian of `exp`.
    fn ljac(&self) -> Self::Jacobian;
    /// Computes the inverse of the right Jacobian of the Lie group element `self`.
    ///
    /// **Note**
    ///
    /// This function implements Eq. (144).
    ///
    /// See also:
    ///
    /// * [Eq. (144)](link-to-equation-144) for the inverse of the right Jacobian.
    /// * [rjac](link-to-rjac) for the right Jacobian.
    fn rjacinv(&self) -> Self::Jacobian;
    /// Computes the inverse of the left Jacobian of the Lie group element `self`.
    ///
    /// **Note**
    ///
    /// This function implements Eq. (146).
    ///
    /// See also:
    ///
    /// * [Eq. (146)](link-to-equation-146) for the inverse of the left Jacobian.
    /// * [ljac](link-to-ljac) for the left Jacobian.
    fn ljacinv(&self) -> Self::Jacobian;

    /// See also: [rplus](link-to-rplus).
    fn plus_j(
        &self,
        t: Self,
        J_mout_ta: Option<&mut Self::Jacobian>,
        J_mout_tb: Option<&mut Self::Jacobian>,
    ) -> Self {
        if let Some(J_mout_ta) = J_mout_ta {
            J_mout_ta.clone_from(&(Self::Jacobian::manif_identity()))
        }
        if let Some(J_mout_tb) = J_mout_tb {
            J_mout_tb.clone_from(&(Self::Jacobian::manif_identity()))
        }
        self.clone() + t
    }
    /// An alias for the right oplus operation.
    fn plus(&self, t: Self) -> Self {
        self.plus_j(t, None, None)
    }
    fn minus_j(
        &self,
        t: Self,
        J_mout_ta: Option<&mut Self::Jacobian>,
        J_mout_tb: Option<&mut Self::Jacobian>,
    ) -> Self {
        if let Some(J_mout_ta) = J_mout_ta {
            J_mout_ta.clone_from(&(Self::Jacobian::manif_identity()))
        }
        if let Some(J_mout_tb) = J_mout_tb {
            J_mout_tb.clone_from(&(Self::Jacobian::from(-Self::Jacobian::manif_identity())))
        }
        self.clone() - t
    }
    fn minus(&self, t: Self) -> Self {
        self.minus_j(t, None, None)
    }
}

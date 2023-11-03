use nalgebra::{UnitQuaternion, Vector3};
use nalgebra_lie::{LieGroupBase, TangentBase};

fn main() {
    let tangent = Vector3::new(0.0, 0.0, 0.1);

    let so3 = tangent.exp_map_j(None);
    println!("so3 {:?}", so3);
    let so3 = tangent.exp_map();
    println!("so3 {:?}", so3);
    // let dim = <UnitQuaternion<f64> as LieGroup<f64, 2, 3, 4>>::DIM;
    // println!("dim {}", dim);
    let t = so3.log_map();
    println!("t {}", t);

    println!("t {}", so3.rplus_j(t, None, None));
}

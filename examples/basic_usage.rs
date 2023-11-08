use manif::{LieGroupBase, TangentBase};
use nalgebra::Vector3;

fn main() {
    let tangent = Vector3::new(0.0, 0.0, 0.1);

    let so3 = tangent.exp_map_j(None);
    println!("so3 {:?}", so3);
    let so3 = tangent.exp_map();
    println!("so3 {:?}", so3);
    let t = so3.log_map();
    println!("t {}", t);

    println!("t {}", so3.rplus_j(t, None, None));
}

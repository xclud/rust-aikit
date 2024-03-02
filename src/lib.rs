use std::ops::Mul;

use image::Rgba32FImage;
use nalgebra::{Matrix3, Vector3};

pub fn warp_into(input: Rgba32FImage, matrix: Matrix3<f32>, output: &mut Rgba32FImage) {
    let inverse = matrix.try_inverse().unwrap();

    let in_width = input.width();
    let in_height = input.height();

    let out_width = output.width();
    let out_height = output.height();

    for out_row in 0..out_width {
        for out_col in 0..out_height {
            let in_pixel = inverse.mul(Vector3::<f32>::new(out_row as f32, out_col as f32, 1f32));

            let in_row = in_pixel[0] as i32;
            let in_col = in_pixel[1] as i32;

            if (0 <= in_row)
                && (in_row < in_width as i32)
                && (0 <= in_col)
                && (in_col < in_height as i32)
            {
                let px = input.get_pixel(in_row as _, in_col as _);

                output[(out_row, out_col)] = *px;
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

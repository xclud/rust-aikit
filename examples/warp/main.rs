use aikit::warp_into;
use image::{Rgb, RgbImage, Rgba32FImage};
use nalgebra::{Matrix3, Vector2};

pub fn read_rgb_image(image_path: &str) -> Rgba32FImage {
    let image = image::open(image_path).unwrap().to_rgba32f();

    return image;
}

fn to_rgb8(image: Rgba32FImage) -> RgbImage {
    let output = RgbImage::from_fn(image.width(), image.height(), |i, j| {
        let px = image.get_pixel(i, j);
        let p = px.0;

        Rgb::<u8>([
            (p[0] * 255.0) as u8,
            (p[1] * 255.0) as u8,
            (p[2] * 255.0) as u8,
        ])
    });

    output
}

fn main() {
    let input = read_rgb_image("t1.jpg");
    let mut output = Rgba32FImage::new(256, 256);

    let mut matrix = Matrix3::identity();

    matrix = matrix.append_translation(&Vector2::new(-100f32, 0f32));

    warp_into(&input, matrix, &mut output);

    _ = to_rgb8(output).save("saved_t1.jpg");
}

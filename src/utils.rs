use rand::Rng;

pub fn draw_input_to_screen(record: Vec<f64>) {
    let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    for (i, &pixel) in record.iter().enumerate() {
        print!(
            "{:<2}",
            chars[(pixel * (chars.len() - 1) as f64).round() as usize]
        );
        if (i + 1) % 28 == 0 {
            println!();
        }
    }
}

pub fn generate_rand_vec(size: u32) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut vec: Vec<f64> = Vec::new();

    for _ in 0..size {
        vec.push(rng.gen_range(-1.0..1.0));
    }

    vec
}

pub fn add_matrices(mat1: &Vec<Vec<f64>>, mat2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    mat1.iter()
        .zip(mat2.iter())
        .map(|(row1, row2)| {
            row1.iter()
                .zip(row2.iter())
                .map(|(val1, val2)| val1 + val2)
                .collect()
        })
        .collect()
}

pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut transposed: Vec<Vec<f64>> = vec![vec![0.0; matrix.len()]; matrix[0].len()];

    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            transposed[j][i] = val;
        }
    }

    transposed
}

pub fn soft_max(vector: &Vec<f64>) -> Vec<f64> {
    let total: f64 = vector.iter().sum();
    let mut soft_maxed: Vec<f64> = Vec::new();

    for value in vector.iter() {
        soft_maxed.push(value / total);
    }
    soft_maxed
}

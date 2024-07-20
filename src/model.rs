use rand::Rng;

#[derive(Debug)]
pub struct Model {
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub layer_sizes: Vec<u32>,
}

impl Model {
    fn generate_rand(size: u32) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut vec: Vec<f64> = Vec::new();

        for _ in 0..size {
            vec.push(rng.gen::<f64>());
        }

        vec
    }

    pub fn new(layer_sizes: Vec<u32>) -> Model {
        let mut biases: Vec<Vec<f64>> = Vec::new();
        let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();

        for biases_size in layer_sizes.iter().skip(1) {
            biases.push(Model::generate_rand(*biases_size));
        }

        for i in 0..layer_sizes.len() - 1 {
            let mut layer_weights: Vec<Vec<f64>> = Vec::new();
            for _ in 0..layer_sizes[i] {
                layer_weights.push(Model::generate_rand(layer_sizes[i + 1]));
            }
            weights.push(layer_weights);
        }

        Model {
            weights,
            biases,
            layer_sizes,
        }
    }

    fn calculate_activation(
        &self,
        activations: &Vec<f64>,
        weights: &Vec<f64>,
        bias: &f64,
        apply_relu: bool,
    ) -> f64 {
        let mut activation = 0.0;

        for i in 0..activations.len() {
            activation += activations[i] * weights[i];
        }

        activation += bias;
        if apply_relu {
            activation = f64::max(0.0, activation); // ReLU
        }

        activation
    }

    fn soft_max(&self, vector: &Vec<f64>) -> Vec<f64> {
        let total: f64 = vector.iter().sum();
        let mut soft_maxed: Vec<f64> = Vec::new();

        for value in vector.iter() {
            soft_maxed.push(value / total);
        }
        soft_maxed
    }

    fn feed_forward(&self, input: Vec<f64>, apply_relu: bool) -> Vec<Vec<f64>> {
        let mut activations: Vec<Vec<f64>> = vec![input];
        for (i, layer_size) in self.layer_sizes.iter().skip(1).enumerate() {
            let weights = self.weights[i].clone();
            let biases = self.biases[i].clone();

            let mut layer_activations: Vec<f64> = Vec::new();
            for j in 0..*layer_size {
                let bias = biases[j as usize];
                let activation_weights = weights[j as usize].clone();
                let activation = self.calculate_activation(
                    &activations[i],
                    &activation_weights,
                    &bias,
                    apply_relu,
                );
                layer_activations.push(activation);
            }
            activations.push(layer_activations);
        }

        activations
    }

    fn gradient_descent_step(
        &self,
        params: &Vec<Vec<f64>>,
        grads: &Vec<Vec<f64>>,
        learning_rate: f64,
    ) -> Vec<Vec<f64>> {
        let mut new_params: Vec<Vec<f64>> = params.clone();
        for (i, grad_layer) in grads.iter().enumerate() {
            for (j, grad) in grad_layer.iter().enumerate() {
                new_params[i][j] -= learning_rate * grad;
            }
        }
        new_params
    }

    fn add_matrices(&self, mat1: &Vec<Vec<f64>>, mat2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

    fn delta(&self, output_layer: &Vec<f64>, label: u8) -> Vec<f64> {
        let mut delta: Vec<f64> = Vec::new();
        for (i, node) in output_layer.iter().enumerate() {
            let desired = if i as u8 == label { 1.0 } else { 0.0 };
            delta.push(node - desired);
        }
        delta
    }

    fn calc_weights_dot(&self, delta: &Vec<f64>, activations: &Vec<f64>) -> Vec<f64> {
        let mut grads: Vec<f64> = Vec::new();
        for node in activations.iter() {
            for desired in delta.iter() {
                grads.push(node * desired);
            }
        }
        grads
    }

    fn transpose(&self, matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut transposed: Vec<Vec<f64>> = vec![vec![0.0; matrix.len()]; matrix[0].len()];

        for (i, row) in matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                transposed[j][i] = val;
            }
        }

        transposed
    }
    fn calc_biases_dot(&self, weights: &Vec<Vec<f64>>, delta: &Vec<f64>) -> Vec<f64> {
        let mut grads: Vec<f64> = Vec::new();
        let weights_t = self.transpose(weights);
        for (_, weight) in weights_t.iter().enumerate() {
            let mut sum = 0.0;
            for (j, node) in weight.iter().enumerate() {
                sum += node * delta[j];
            }
            grads.push(sum);
        }
        grads
    }

    pub fn backprop(&self, input: Vec<f64>, label: u8) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let layers = self.layer_sizes.len();
        let mut grads_w: Vec<Vec<Vec<f64>>> = vec![Vec::new(); layers - 1];
        let mut grads_b: Vec<Vec<f64>> = vec![Vec::new(); layers - 1];

        let activations = self.feed_forward(input.clone(), true);
        println!("Activations: {:?}", activations);
        let zs: Vec<Vec<f64>> = self.feed_forward(input.clone(), false);

        let mut delta = self.delta(&activations[layers - 1], label);
        grads_b[layers - 2] = delta.clone();
        // grads_w[layers - 2] = self.calc_weights_dot(&delta, &activations[layers - 2]);

        for l in (2..(layers)).rev() {
            let z = zs[l - 1].clone();
            let mut new_delta: Vec<f64> = self.calc_biases_dot(&self.weights[l - 1], &delta);
            for i in 0..new_delta.len() {
                // ReLU derivative
                new_delta[i] *= if z[i] > 0.0 { 1.0 } else { 0.0 };
            }

            grads_b[l - 2] = new_delta.clone();
            // grads_w[l - 2] = self.calc_weights_dot(&delta, &activations[l]);

            delta = new_delta;
        }

        (grads_w, grads_b)
    }

    pub fn update_mini_batch(mut self, inputs: Vec<Vec<f64>>, labels: Vec<u8>, learning_rate: f64) {
        let mut grads_w: Vec<Vec<Vec<f64>>> = vec![Vec::new(); self.weights.len()];
        let mut grads_b: Vec<Vec<f64>> = vec![Vec::new(); self.biases.len()];

        for i in 0..inputs.len() {
            let input = inputs[i].clone();
            let label = labels[i];
            let (delta_grads_w, delta_grads_b) = self.backprop(input, label);
            // grads_w = self.add_matrices(&grads_w, &delta_grads_w);
            grads_b = self.add_matrices(&grads_b, &delta_grads_b);
        }

        self.biases = self.gradient_descent_step(&self.biases, &grads_b, learning_rate);
        // self.weights = self.gradient_descent_step(&self.weights, &grads_w, learning_rate);
    }

    pub fn predict(&self, input: Vec<f64>) -> (u8, f64) {
        let activations = self.feed_forward(input, true);
        let output_layer = self.soft_max(&activations.last().unwrap());

        let mut label: u8 = 0;
        let mut score: f64 = f64::MIN;
        for (i, output_node) in output_layer.iter().enumerate() {
            if output_node > &score {
                label = i.try_into().unwrap();
                score = *output_node;
            }
        }

        (label, score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backprop() {
        let mut model: Model = Model::new(vec![2, 3, 2]);
        let input = vec![1.0, 2.0];
        model.weights = vec![
            vec![vec![0.2, 0.4], vec![0.6, 0.8], vec![0.5, 0.1]], // 2 → 3
            vec![vec![0.1, 0.3, 0.5], vec![0.2, 0.4, 0.6]],       // 3 → 2
        ];
        model.biases = vec![
            vec![0.1, 0.2, 0.3], // 2 → 3
            vec![0.1, 0.2],      // 3 → 2
        ];

        let (grads_w, grads_b) = model.backprop(input, 1);

        assert_eq!(
            grads_b,
            vec![
                vec![0.3390000000000001, 0.8210000000000002, 1.3030000000000002], // 2 → 3
                vec![1.4300000000000002, 0.9800000000000002] // 3 → 2
            ]
        );

        //        assert_eq!(
        //            grads_w,
        //            vec![
        //                vec![0.379, 0.758, 0.909, 1.818, 1.439, 2.878], // 2 → 3
        //                vec![2.869, 3.624, 1.51, 2.166, 2.736, 1.14]    // 3 → 2
        //            ]
        //        );
    }
}

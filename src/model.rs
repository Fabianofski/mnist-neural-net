use rand::Rng;

#[derive(Debug)]
pub struct Model {
    pub weights: Vec<Vec<f64>>,
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
        let mut weights: Vec<Vec<f64>> = Vec::new();

        for biases_size in layer_sizes.iter().skip(1) {
            biases.push(Model::generate_rand(*biases_size));
        }

        for i in 0..layer_sizes.len() - 1 {
            let weights_size = layer_sizes[i] * layer_sizes[i + 1];
            weights.push(Model::generate_rand(weights_size));
        }

        Model {
            weights,
            biases,
            layer_sizes,
        }
    }

    fn calculate_activation(&self, activations: &Vec<f64>, weights: &Vec<f64>, bias: &f64, apply_relu: bool) -> f64 {
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
                let activation = self.calculate_activation(&activations[i], &weights, &bias, apply_relu);
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
            delta.push(desired - node);
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

    fn backprop(&self, input: Vec<f64>, label: u8) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let layers = self.layer_sizes.len();
        let mut grads_w: Vec<Vec<f64>> = vec![Vec::new(); layers];
        let mut grads_b: Vec<Vec<f64>> = vec![Vec::new(); layers];

        let activations = self.feed_forward(input.clone(), true);
        let zs: Vec<Vec<f64>> = self.feed_forward(input.clone(), false);
        
        let mut delta = self.delta(&activations[layers - 1], label); 
        grads_b[layers - 1] = delta.clone();
        grads_w[layers - 1] = self.calc_weights_dot(&delta, &activations[layers - 2]);

        for l in 2..layers {
            let z = zs[layers - l].clone();
            let new_delta = self.calc_weights_dot(&delta, &self.weights[layers - l]);
            let mut z_delta: Vec<f64> = Vec::new();
            for (i, desired) in new_delta.iter().enumerate() {
                z_delta.push(desired * z[i]);
            }

            grads_b[layers - l] = z_delta.clone();
            grads_w[layers - l] = self.calc_weights_dot(&z_delta, &activations[layers - l - 1]);

            delta = z_delta.clone();
        }

        (grads_w, grads_b)
    }

    pub fn update_mini_batch(mut self, inputs: Vec<Vec<f64>>, labels: Vec<u8>, learning_rate: f64) {
        let mut grads_w: Vec<Vec<f64>> = vec![Vec::new(); self.weights.len()];
        let mut grads_b: Vec<Vec<f64>> = vec![Vec::new(); self.biases.len()];

        for i in 0..inputs.len() {
            let input = inputs[i].clone();
            let label = labels[i];
            let (delta_grads_w, delta_grads_b) = self.backprop(input, label);
            grads_w = self.add_matrices(&grads_w, &delta_grads_w);
            grads_b = self.add_matrices(&grads_b, &delta_grads_b);
        }

        self.biases = self.gradient_descent_step(&self.biases, &grads_b, learning_rate);
        self.weights = self.gradient_descent_step(&self.weights, &grads_w, learning_rate);
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

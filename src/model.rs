use rand::Rng;

#[derive(Debug)]
pub struct Model {
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
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
        let biases_size: u32 = layer_sizes.iter().skip(1).sum();
        let mut weights_size: u32 = 0;
        for i in 0..layer_sizes.len() - 1 {
            weights_size += layer_sizes[i] * layer_sizes[i + 1];
        }

        println!("Weights: {}, Biases: {}", weights_size, biases_size);
        let weights = Model::generate_rand(weights_size);
        let biases = Model::generate_rand(biases_size);
        Model {
            weights,
            biases,
            layer_sizes,
        }
    }

    fn calculate_activation(&self, activations: &Vec<f64>, weights: &[f64], bias: &f64) -> f64 {
        let mut activation = 0.0;

        for i in 0..activations.len() {
            activation += activations[i] * weights[0];
        }

        activation += bias;
        activation = f64::max(0.0, activation); // ReLU

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

    pub fn predict(&self, pixels: Vec<f64>) -> (u8, f64) {
        let mut previous_activations: Vec<f64> = pixels;
        let mut weight_idx = 0;
        let mut bias_idx = 0;
        for layer_size in self.layer_sizes.iter().skip(1) {
            let mut activations: Vec<f64> = Vec::new();
            for _ in 0..*layer_size {
                let weight_end = weight_idx + &previous_activations.len();
                let weights = &self.weights[weight_idx..weight_end];
                let bias = &self.biases[bias_idx];

                let activation = self.calculate_activation(&previous_activations, weights, bias);
                activations.push(activation);

                weight_idx = weight_end;
                bias_idx += 1;
            }
            previous_activations = activations;
        }

        let output_layer = self.soft_max(&previous_activations);

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

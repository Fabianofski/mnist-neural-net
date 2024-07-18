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

    fn calculate_activation(&self, activations: &Vec<f64>, weights: &Vec<f64>, bias: &f64) -> f64 {
        let mut activation = 0.0;

        for i in 0..activations.len() {
            activation += activations[i] * weights[i];
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
        for (i, layer_size) in self.layer_sizes.iter().skip(1).enumerate() {
            let weights = self.weights[i].clone();
            let biases = self.biases[i].clone();

            let mut activations: Vec<f64> = Vec::new();
            for j in 0..*layer_size {
                let bias = biases[j as usize];
                let activation = self.calculate_activation(&previous_activations, &weights, &bias);
                activations.push(activation);
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

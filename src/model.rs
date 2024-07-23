use crate::utils;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, Read, Write};

#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    pub layer_sizes: Vec<u32>,
}

impl Model {
    pub fn new(layer_sizes: Vec<u32>) -> Model {
        let mut biases: Vec<Vec<f64>> = Vec::new();
        let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();

        for biases_size in layer_sizes.iter().skip(1) {
            biases.push(vec![0.0; *biases_size as usize]);
        }

        for i in 1..layer_sizes.len() {
            let mut layer_weights: Vec<Vec<f64>> = Vec::new();
            for _ in 0..layer_sizes[i] {
                layer_weights.push(utils::generate_rand_vec(layer_sizes[i - 1]));
            }
            weights.push(layer_weights);
        }

        Model {
            weights,
            biases,
            layer_sizes,
        }
    }

    pub fn save(&self, filename: &str) -> io::Result<()> {
        let json = serde_json::to_string(self)?;
        let mut file = File::create(filename)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn load(filename: &str) -> io::Result<Self> {
        let mut file = File::open(filename)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        let data: Model = serde_json::from_str(&json)?;
        Ok(data)
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

    fn feed_forward(&self, input: Vec<f64>, apply_relu: bool) -> Vec<Vec<f64>> {
        let mut activations: Vec<Vec<f64>> = vec![input];
        for (i, layer_size) in self.layer_sizes.iter().skip(1).enumerate() {
            let weights = self.weights[i].clone();
            let biases = self.biases[i].clone();

            let mut layer_activations: Vec<f64> = Vec::new();
            for j in 0..*layer_size as usize {
                let bias = biases[j];
                let activation_weights = weights[j].clone();
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

    fn delta(output_layer: &Vec<f64>, label: u8) -> Vec<f64> {
        let mut delta: Vec<f64> = Vec::new();
        for (i, node) in output_layer.iter().enumerate() {
            let desired = if i as u8 == label { 1.0 } else { 0.0 };
            delta.push(node - desired);
        }
        delta
    }

    fn calc_weights_dot(delta: &Vec<f64>, activations: &Vec<f64>) -> Vec<Vec<f64>> {
        let mut grads: Vec<Vec<f64>> = Vec::new();
        for desired in delta.iter() {
            let mut grad: Vec<f64> = Vec::new();
            for node in activations.iter() {
                grad.push(node * desired);
            }
            grads.push(grad);
        }
        grads
    }

    fn calc_biases_dot(weights: &Vec<Vec<f64>>, delta: &Vec<f64>) -> Vec<f64> {
        let mut grads: Vec<f64> = Vec::new();
        let weights_t = utils::transpose(weights);
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
        let layers = self.layer_sizes.len() - 1;
        let mut grads_w: Vec<Vec<Vec<f64>>> = vec![Vec::new(); layers];
        let mut grads_b: Vec<Vec<f64>> = vec![Vec::new(); layers];

        let activations = self.feed_forward(input.clone(), true);
        let zs: Vec<Vec<f64>> = self.feed_forward(input.clone(), false);

        let mut delta = Model::delta(&activations[layers], label);

        for l in (0..layers).rev() {
            grads_b[l] = delta.clone();
            grads_w[l] = Model::calc_weights_dot(&delta, &activations[l]);

            let z = zs[l].clone();
            delta = Model::calc_biases_dot(&self.weights[l], &delta);
            for i in 0..delta.len() {
                // ReLU derivative
                delta[i] *= if z[i] > 0.0 { 1.0 } else { 0.0 };
            }
        }

        (grads_w, grads_b)
    }

    fn gradient_descent_step(
        params: &Vec<Vec<f64>>,
        grads: &Vec<Vec<f64>>,
        batch_size: f64,
        learning_rate: f64,
    ) -> Vec<Vec<f64>> {
        let mut new_params: Vec<Vec<f64>> = params.clone();
        for (i, grad_layer) in grads.iter().enumerate() {
            for (j, grad) in grad_layer.iter().enumerate() {
                new_params[i][j] -= learning_rate * (grad / batch_size);
            }
        }
        new_params
    }

    pub fn update_mini_batch(&mut self, inputs: Vec<(u8, Vec<f64>)>, learning_rate: f64) {
        let mut grads_w: Vec<Vec<Vec<f64>>> = self
            .weights
            .iter()
            .map(|layer| layer.iter().map(|neuron| vec![0.0; neuron.len()]).collect())
            .collect();
        let mut grads_b: Vec<Vec<f64>> = self
            .biases
            .iter()
            .map(|layer| vec![0.0; layer.len()])
            .collect();

        for i in 0..inputs.len() {
            let (label, input) = inputs[i].clone();
            let (delta_grads_w, delta_grads_b) = self.backprop(input, label);
            for (i, grad_layer) in delta_grads_w.iter().enumerate() {
                grads_w[i] = utils::add_matrices(&grads_w[i], grad_layer);
            }
            grads_b = utils::add_matrices(&grads_b, &delta_grads_b);
        }

        let batch_size = inputs.len() as f64;
        self.biases =
            Model::gradient_descent_step(&self.biases, &grads_b, batch_size, learning_rate);
        for (i, grad_layer) in grads_w.iter().enumerate() {
            self.weights[i] = Model::gradient_descent_step(
                &self.weights[i],
                &grad_layer,
                batch_size,
                learning_rate,
            );
        }
    }

    pub fn predict(&self, input: Vec<f64>) -> (u8, f64) {
        utils::draw_input_to_screen(input.clone());

        let activations = self.feed_forward(input, true);
        let output_layer = utils::soft_max(&activations.last().unwrap());

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

    pub fn check_accuracy(&self, inputs: Vec<(u8, Vec<f64>)>) -> f64 {
        let mut correct = 0.0;
        let total = inputs.len() as f64;

        for (label, input) in inputs {
            let (pred_label, _) = self.predict(input);
            if pred_label == label {
                correct += 1.0;
            }
        }

        correct / total
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
                vec![1.4300000000000002, 0.9800000000000002]                      // 3 → 2
            ]
        );

        assert_eq!(
            grads_w,
            vec![
                vec![
                    vec![0.3390000000000001, 0.6780000000000002],
                    vec![0.8210000000000002, 1.6420000000000003],
                    vec![1.3030000000000002, 2.6060000000000003]
                ], // 2 → 3
                vec![
                    vec![1.5730000000000004, 3.432000000000001, 1.4300000000000002],
                    vec![1.0780000000000003, 2.3520000000000008, 0.9800000000000002]
                ] // 3 → 2
            ]
        );
    }
}

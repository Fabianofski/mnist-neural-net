mod model;
mod utils;

use std::usize;

use csv::StringRecord;
use indicatif::ProgressIterator;
use model::Model;
use rand::seq::SliceRandom;
use rand::thread_rng;


fn record_to_data(record: StringRecord) -> (u8, Vec<f64>) {
    let label = record[0].parse::<u8>().unwrap();
    let pixels = record
        .iter()
        .skip(1)
        .map(|x| x.parse::<f64>().unwrap() / 255.0)
        .collect();
    (label, pixels)
}

fn load_data_from_csv(path: &str) -> Vec<(u8, Vec<f64>)> {
    let mut inputs: Vec<(u8, Vec<f64>)> = Vec::new();

    let mut rdr = csv::Reader::from_path(path).unwrap();
    for record in rdr.records() {
        let data = record.unwrap();
        let input = record_to_data(data);
        inputs.push(input);
    }

    inputs
}

fn train_model(
    model: &mut Model,
    inputs_train: Vec<(u8, Vec<f64>)>,
    batch_size: usize,
    learning_rate: f64,
    epochs: u8,
) {
    for epoch in 1..=epochs {
        println!("Epoch: {}", epoch);

        let mut shuffled = inputs_train.clone();
        shuffled.shuffle(&mut thread_rng());

        for batch in (0..shuffled.len()).step_by(batch_size).progress() {
            let end = usize::min(batch + batch_size, shuffled.len() - 1);
            let inputs: Vec<(u8, Vec<f64>)> = shuffled[batch..end].to_vec();
            model.update_mini_batch(inputs, learning_rate);
        }
        let accuracy = model.check_accuracy(inputs_train.clone());
        println!("Train Accuracy: {:.2}%\n", accuracy * 100.0);
        model
            .save(format!("models/model-{}.json", epoch).as_str())
            .unwrap();
    }
}

fn training_cycle(inputs_train: Vec<(u8, Vec<f64>)>, inputs_test: Vec<(u8, Vec<f64>)>) {
    let mut model: Model = Model::new(vec![784, 16, 16, 10]);

    let batch_size: usize = 64;
    let epochs = 12;
    let learning_rate = 0.01;

    train_model(&mut model, inputs_train, batch_size, learning_rate, epochs);

    let accuracy = model.check_accuracy(inputs_test.clone());
    println!("Test Accuracy {:.2}\n", accuracy * 100.0);
}

fn main() {
    let mut inputs_test = load_data_from_csv("src/mnist_test.csv");
    let training: bool = false;

    if training {
        let inputs_train = load_data_from_csv("src/mnist_train.csv");
        training_cycle(inputs_train, inputs_test);
    } else {
        let model: Model = Model::load("models/model-12.json").unwrap();
        inputs_test.shuffle(&mut thread_rng());
        let (label, pixels) = inputs_test.first().unwrap();
        let (pred_label, pred_score) = model.predict(pixels.clone(), true);
        println!("Prediction: {}, {:.2}", pred_label, pred_score);
        println!("Label: {}", label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_to_data() {
        let vec = vec!["3", "255", "127.5", "255", "63.75"];
        let record: StringRecord = StringRecord::from(vec);

        let (label, pixels) = record_to_data(record);

        assert_eq!(label, 3);
        assert_eq!(pixels, vec![1.0, 0.5, 1.0, 0.25]);
    }
}

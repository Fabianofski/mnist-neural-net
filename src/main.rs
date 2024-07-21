mod model;

use std::usize;

use csv::StringRecord;
use model::Model;
use rand::thread_rng;
use rand::seq::SliceRandom;
use indicatif::ProgressIterator;

fn save_as_image(label: u8, record: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = image::ImageBuffer::new(28, 28);

    for (i, &pixel) in record.iter().enumerate() {
        let x = (i) % 28;
        let y = (i) / 28;
        img.put_pixel(x as u32, y as u32, image::Luma([pixel]));
    }

    let name = format!("label-{}.png", label);
    img.save(name)?;

    Ok(())
}

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
    epochs: u8,
) {

    for epoch in 1..=epochs {
        println!("Starting Epoch: {}", epoch);

        let mut shuffled = inputs_train.clone();
        shuffled.shuffle(&mut thread_rng());

        for batch in (0..shuffled.len()).step_by(batch_size).progress() {
            let end = usize::min(batch+32, shuffled.len() - 1);
            let inputs: Vec<(u8, Vec<f64>)> = shuffled[batch..end].to_vec();
            model.update_mini_batch(inputs, 0.1);
            println!("Weights: {:?}", model.weights[2][0]);
        }
        let accuracy = model.check_accuracy(inputs_train.clone());
        println!("Train Accuracy: {:2}%\n", accuracy);
    }
}

fn main() {
    let mut model: Model = Model::new(vec![784, 16, 16, 10]);

    let inputs_train = load_data_from_csv("src/mnist_train.csv");
    // let inputs_val = load_data_from_csv("src/mnist_val.csv");
    let inputs_test = load_data_from_csv("src/mnist_test.csv");

    let batch_size: usize = 32;
    let epochs = 12;

    train_model(&mut model, inputs_train, batch_size, epochs);

    let accuracy = model.check_accuracy(inputs_test.clone());
    println!("Test Accuracy: {:2}%\n", accuracy);
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

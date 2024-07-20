mod model;

use csv::StringRecord;
use model::Model;

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

fn main() {
    let model: Model = Model::new(vec![784, 16, 16, 10]);
    let batch_size = 1;
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    let mut rdr = csv::Reader::from_path("src/mnist_test.csv").unwrap();
    for (i, record) in rdr.records().enumerate() {
        let data = record.unwrap();
        let (label, pixels) = record_to_data(data);

        inputs.push(pixels);
        labels.push(label);

        if i == batch_size - 1 {
            break;
        }
    }

    model.update_mini_batch(inputs, labels, 0.0001);
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

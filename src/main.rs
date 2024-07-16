use csv::StringRecord; 

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

fn record_to_data(record: StringRecord) -> (u8, Vec<u8>) {
    let label = record[0].parse::<u8>().unwrap();
    let mut pixels: Vec<u8> = Vec::new(); 

    for (i, pixel) in record.iter().enumerate() {
        if i == 0 {
            continue;
        }

        let pixel = pixel.parse::<u8>().unwrap();
        pixels.push(pixel);
    }

    (label, pixels)
}

fn main(){
    let mut rdr = csv::Reader::from_path("src/mnist_test.csv").unwrap();
    let record = rdr.records().next().unwrap();

    let record = record.unwrap();
    let (label, pixels) = record_to_data(record);

    save_as_image(label, pixels).unwrap();
}

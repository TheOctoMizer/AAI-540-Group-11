use anyhow::{Context, Result};
use std::time::Instant;
use tract_onnx::prelude::*;

pub struct Autoencoder {
    model: TypedRunnableModel<TypedModel>,
    input_size: usize,
    output_size: usize,
}

impl Autoencoder {
    pub fn new(model_path: &str) -> Result<Self> {
        let start_time = Instant::now();
        
        log::info!("Loading autoencoder model from: {}", model_path);

        let input_size = 77;

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("Failed to load autoencoder ONNX model")?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, input_size)),
            )
            .context("Failed to set autoencoder input fact")?
            .into_optimized()
            .context("Failed to optimize autoencoder model")?
            .into_runnable()
            .context("Failed to make autoencoder model runnable")?;

        // Infer output size from a dummy run. Some autoencoders reconstruct the full input
        // rather than outputting a bottleneck embedding.
        let dummy = vec![0f32; input_size];
        let dummy_array = tract_ndarray::Array2::from_shape_vec((1, input_size), dummy)
            .context("Failed to build dummy autoencoder input")?;
        let dummy_tensor: Tensor = dummy_array.into();
        let dummy_outputs = model
            .run(tvec!(dummy_tensor.into()))
            .context("Failed to run dummy autoencoder inference")?;
        let dummy_out = dummy_outputs[0]
            .to_array_view::<f32>()
            .context("Failed to view dummy autoencoder output as f32")?;
        let output_size = dummy_out.len();

        let load_time = start_time.elapsed();
        log::info!("Autoencoder model loaded successfully in {:?}", load_time);
        
        Ok(Self { 
            model,
            input_size,
            output_size,
        })
    }
    
    pub fn encode(&self, input: Vec<f32>) -> Result<Vec<f32>> {
        let start_time = Instant::now();
        
        log::debug!("Encoding input with {} features", input.len());

        if input.len() != self.input_size {
            anyhow::bail!(
                "Autoencoder expected {} features, got {}",
                self.input_size,
                input.len()
            );
        }

        let input_array = tract_ndarray::Array2::from_shape_vec((1, self.input_size), input)
            .context("Failed to build autoencoder input array")?;
        let input_tensor: Tensor = input_array.into();

        let outputs = self
            .model
            .run(tvec!(input_tensor.into()))
            .context("Failed to run autoencoder inference")?;

        // Expect output shape [1, output_size]
        let out = outputs[0]
            .to_array_view::<f32>()
            .context("Failed to view autoencoder output as f32")?;

        let encoded: Vec<f32> = out.iter().copied().collect();

        if encoded.len() != self.output_size {
            log::warn!(
                "Autoencoder output length mismatch: expected {}, got {}",
                self.output_size,
                encoded.len()
            );
        }
        
        let inference_time = start_time.elapsed();
        log::debug!("Autoencoder inference completed in {:?}", inference_time);
        
        Ok(encoded)
    }
    
    pub fn get_input_shape(&self) -> Result<Vec<usize>> {
        Ok(vec![1, self.input_size])
    }
    
    pub fn get_output_shape(&self) -> Result<Vec<usize>> {
        Ok(vec![1, self.output_size])
    }
}

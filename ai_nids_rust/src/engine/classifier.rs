use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tract_onnx::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub confidence: f32,
    pub class_id: usize,
}

pub struct XGBoostClassifier {
    model: TypedRunnableModel<TypedModel>,
    label_map: HashMap<usize, String>,
    num_classes: usize,
    input_size: usize,
}

impl XGBoostClassifier {
    fn parse_outputs(&self, outputs: &TVec<TValue>) -> Result<(usize, f32, Option<Vec<f32>>)> {
        // XGBoost ONNX exports vary:
        // - output[0] can be the predicted class id (i64)
        // - output[1] can be probabilities/scores (f32/f64)
        // - some models output probabilities directly as output[0]
        let predicted_class_id: Option<usize> = outputs
            .get(0)
            .and_then(|t| t.to_array_view::<i64>().ok())
            .and_then(|view| view.iter().next().copied())
            .and_then(|v| usize::try_from(v).ok());

        let probabilities: Option<Vec<f32>> = if outputs.len() >= 2 {
            let t = &outputs[1];
            match t.to_array_view::<f32>() {
                Ok(view) => Some(view.iter().copied().collect()),
                Err(_) => match t.to_array_view::<f64>() {
                    Ok(view) => Some(view.iter().map(|v| *v as f32).collect()),
                    Err(_) => None,
                },
            }
        } else {
            let t = &outputs[0];
            match t.to_array_view::<f32>() {
                Ok(view) => Some(view.iter().copied().collect()),
                Err(_) => match t.to_array_view::<f64>() {
                    Ok(view) => Some(view.iter().map(|v| *v as f32).collect()),
                    Err(_) => None,
                },
            }
        };

        let (class_id, confidence) = if let Some(probs) = probabilities.as_ref() {
            let (cid, &conf) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .context("Failed to find max probability")?;
            (cid, conf)
        } else if let Some(cid) = predicted_class_id {
            (cid, 1.0)
        } else {
            anyhow::bail!(
                "Classifier output not understood (neither probabilities nor predicted class id)"
            );
        };

        Ok((class_id, confidence, probabilities))
    }

    pub fn new(model_path: &str, label_map_path: &str, input_size: usize) -> Result<Self> {
        let start_time = Instant::now();
        
        log::info!("Loading XGBoost classifier model from: {}", model_path);
        log::info!("Loading label map from: {}", label_map_path);
        
        // Load label map
        let label_map_content = std::fs::read_to_string(label_map_path)
            .context("Failed to read label map file")?;
        
        let label_map: HashMap<usize, String> = serde_json::from_str(&label_map_content)
            .context("Failed to parse label map JSON")?;
        
        let num_classes = label_map.len();

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("Failed to load classifier ONNX model")?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, input_size)))
            .context("Failed to set classifier input fact")?
            .into_optimized()
            .context("Failed to optimize classifier model")?
            .into_runnable()
            .context("Failed to make classifier model runnable")?;
        
        let load_time = start_time.elapsed();
        log::info!("XGBoost classifier model loaded successfully in {:?}", load_time);
        log::info!("Loaded {} class labels", num_classes);
        
        Ok(Self { 
            model,
            label_map,
            num_classes,
            input_size,
        })
    }
    
    pub fn classify(&self, input: Vec<f32>) -> Result<ClassificationResult> {
        let start_time = Instant::now();
        
        log::debug!("Classifying input with {} features", input.len());

        if input.len() != self.input_size {
            anyhow::bail!(
                "Classifier expected {} features, got {}",
                self.input_size,
                input.len()
            );
        }

        let input_array = tract_ndarray::Array2::from_shape_vec((1, self.input_size), input)
            .context("Failed to build classifier input array")?;
        let input_tensor: Tensor = input_array.into();

        let outputs = self
            .model
            .run(tvec!(input_tensor.into()))
            .context("Failed to run classifier inference")?;

        let (class_id, confidence, _probabilities) =
            self.parse_outputs(&outputs).context("Failed to parse classifier outputs")?;
        
        let label = self.label_map
            .get(&class_id)
            .cloned()
            .unwrap_or_else(|| format!("Class_{}", class_id));
        
        let inference_time = start_time.elapsed();
        log::debug!("XGBoost classification completed in {:?}", inference_time);
        log::debug!(
            "Predicted: {} (class_id: {}, confidence: {:.4})",
            label,
            class_id,
            confidence
        );
        
        Ok(ClassificationResult {
            label,
            confidence,
            class_id,
        })
    }
    
    pub fn classify_with_probabilities(&self, input: Vec<f32>) -> Result<(ClassificationResult, Vec<f32>)> {
        if input.len() != self.input_size {
            anyhow::bail!(
                "Classifier expected {} features, got {}",
                self.input_size,
                input.len()
            );
        }

        let input_array = tract_ndarray::Array2::from_shape_vec((1, self.input_size), input)
            .context("Failed to build classifier input array")?;
        let input_tensor: Tensor = input_array.into();

        let outputs = self
            .model
            .run(tvec!(input_tensor.into()))
            .context("Failed to run classifier inference")?;

        let (class_id, confidence, probabilities) =
            self.parse_outputs(&outputs).context("Failed to parse classifier outputs")?;

        let probabilities = probabilities.unwrap_or_else(|| {
            let mut one_hot = vec![0f32; self.num_classes];
            if class_id < one_hot.len() {
                one_hot[class_id] = 1.0;
            }
            one_hot
        });

        let label = self
            .label_map
            .get(&class_id)
            .cloned()
            .unwrap_or_else(|| format!("Class_{}", class_id));

        let result = ClassificationResult {
            label,
            confidence,
            class_id,
        };

        Ok((result, probabilities))
    }
    
    pub fn get_input_shape(&self) -> Result<Vec<usize>> {
        Ok(vec![1, self.input_size])
    }
    
    pub fn get_num_classes(&self) -> usize {
        self.num_classes
    }
    
    pub fn get_class_labels(&self) -> Vec<String> {
        let mut labels = vec![String::new(); self.label_map.len()];
        for (id, label) in &self.label_map {
            if *id < labels.len() {
                labels[*id] = label.clone();
            }
        }
        labels
    }
}

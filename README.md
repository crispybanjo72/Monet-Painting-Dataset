# Monet-Painting-Dataset
Monet Painting Dataset

Monet GAN: CycleGAN-Based Artistic Style Transfer
Kaggle Code Competition Report
Author: Carson
Date: September 29, 2025

1. Problem Description
The Monet GAN challenge on Kaggle is a code-based competition focused on generative deep learning. The objective is to build a model that transforms real-world photographs into Monet-style paintings using unpaired image-to-image translation. Participants must submit a zip file (images.zip) containing 7,000–10,000 stylized images, each sized 256×256 pixels. The evaluation metric is based on how convincingly the generated images resemble Monet’s artistic style.
To solve this, we use CycleGAN—a generative adversarial network architecture designed for unpaired domain translation. CycleGAN learns bidirectional mappings between two domains (photos and Monet paintings) using two generators and two discriminators. It enforces cycle consistency, meaning a photo translated to Monet and then back should resemble the original photo.

2. Data Overview
The dataset provided by Kaggle includes:
- Monet paintings: 1,072 JPEG images in monet_jpg/
- Real-world photos: 6,287 JPEG images in photo_jpg/
- Image format: JPEG
- Image dimensions: All images are resized to 256×256 pixels with 3 RGB channels
- Structure: Flat folder of images, no labels or metadata
This setup is ideal for unsupervised learning, where the goal is to learn style transfer without paired examples.

3. Exploratory Data Analysis (EDA)
EDA was minimal due to the unstructured nature of the dataset. Key observations:
- Monet images exhibit high color saturation, brushstroke texture, and impressionist composition.
- Photo images vary widely in content (landscapes, architecture, nature) but are consistently realistic.
- All images were resized to 256×256 and normalized to [-1, 1] for model input.
No additional preprocessing (e.g., cropping, augmentation) was applied to preserve style fidelity.

4. Model Architecture
We implemented a standard CycleGAN architecture using TensorFlow:
- Generators (G and F):
- Downsample blocks: Conv2D → InstanceNorm → LeakyReLU
- Upsample blocks: Conv2DTranspose → InstanceNorm → ReLU
- Final layer: Conv2DTranspose with tanh activation
- Input/output shape: (256, 256, 3)
- Discriminators (DX and DY):
- PatchGAN-style Conv2D blocks
- Output: Real/fake prediction map
- Custom Layers:
- InstanceNormalization implemented manually (no tensorflow_addons)
All models were built modularly for reproducibility and future extension.

5. Training Strategy
Due to Kaggle’s 5-hour GPU runtime limit, we used a lightweight training strategy:
- Epochs: 5
- Training pairs: 300 Monet + 300 Photo images
- Loss functions:
- Adversarial loss (BinaryCrossentropy)
- Cycle consistency loss (L1)
- Identity loss (L1)
- Optimizers: Adam with learning rate 2e-4 and β₁ = 0.5
- Batch size: 1 (standard for CycleGAN)
Training was monitored with step-wise logging every 50 iterations.

6. Results
After training, we used the generator (Photo → Monet) to stylize 7,000 photo images:
- Output format: JPEG
- Size: 256×256 pixels
- Directory: generated_images/
- Submission file: images.zip containing all stylized images
The model successfully generated a compliant submission file. Visual inspection showed clear stylistic transformation, though fidelity can be improved with longer training or pretrained weights.

7. Discussion & Conclusion
This project demonstrates the effectiveness of CycleGAN for artistic style transfer without paired data. Despite limited training time, the model produced visually plausible Monet-style images suitable for leaderboard submission. Key takeaways:
- CycleGAN is robust for unpaired domain translation
- InstanceNormalization is critical for style fidelity
- GPU runtime constraints require strategic dataset sampling
- Modular architecture enables rapid iteration and future fine-tuning
Future improvements could include:
- Training on full datasets with checkpointing
- Incorporating perceptual loss or style-aware discriminators
- Using pretrained generators for faster inference-only pipelines
This notebook serves as both a reproducible baseline and a submission-ready pipeline for the Monet GAN competition.

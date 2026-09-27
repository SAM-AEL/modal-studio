# HOW TO TRAIN: Flux LoRA Workflow

This guide covers the end-to-end process of training a Flux.1 LoRA using this studio.

## 1. Dataset Preparation

A high-quality dataset is critical for good results.

- **Images**: Aim for 10-20 high-quality PNG/JPG images of your subject.
- **Renaming**: Use the **"Rename Files (001...)"** button in the Dataset Library to standardize filenames. This helps the training script process them reliably.
- **Captioning**: Use the **"Batch Caption All"** button to apply a base caption to all images. 
    - *Tip*: Include your `triggerWord` in every caption (e.g., "a photo of my_trigger_word in a forest").
- **Verification**: Ensure your images and captions are synced to the Modal Volume by checking the Dataset view status.

## 2. Setting up Volumes (Reliability Analysis)

The system uses two persistent Modal Volumes:
1. `lora-datasets`: Stores your images and metadata.
2. `lora-outputs`: Stores the trained weights (`.safetensors`).

> [!IMPORTANT]
> Volumes are automatically committed after writes. If training fails due to "Missing Dataset", ensure you have saved images to the correct dataset name in the Studio or uploaded them to the `lora-datasets` volume.

## 3. Training Parameters

- **Trigger Word**: A unique identifier for your concept (e.g., `sks`, `my_trigger_word`).
- **Steps**: 1000 steps is a good starting point for 15-20 images.
- **Rank**: 16-32 is standard for Flux. Higher rank captures more detail but increases file size.
- **Learning Rate**: `1e-4` is generally stable.

## 4. Launching & Monitoring

1. Go to the **Train** tab.
2. Select your dataset and enter your trigger word.
3. Click **Start Training**.
4. Monitor the **Training Console**. Logs are streamed in real-time from the A100 GPU instance.
5. If you need to abort, use the **"Stop Training"** button. This will immediately terminate the Modal task and stop GPU billing.
6. If you see OOM (Out of Memory) errors, try reducing the resolution or batch size.

## 5. Downloading Results

Once training is complete:
- The status will change to **"Training Complete ✓"**.
- A **"Download LoRA"** button will appear in the header.
- You can also find your trained models in the **"Ready LoRAs"** section at the bottom of the Train view.

---
*Note: Training runs on A100 GPUs on Modal. Ensure your Modal account has sufficient credits.*

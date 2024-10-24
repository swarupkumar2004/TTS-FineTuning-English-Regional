# TTS-FineTuning-English-Regional

This project demonstrates fine-tuning two TTS models: one for English technical jargon and another for a regional language. It includes model optimization techniques for fast inference.

## Project Structure

- `datasets/`: Contains datasets for English technical terms and regional language.
- `src/`: Contains Python scripts for model fine-tuning, utilities, and inference optimization.
- `models/`: Stores pre-trained and fine-tuned models.
- `requirements.txt`: Dependencies required for running the project.

## Instructions

1. Prepare datasets in `datasets/`.
2. Fine-tune the models by running `src/english_tts.py` and `src/regional_tts.py`.
3. Optionally, optimize the model for fast inference using `src/inference_optimization.py`.
4. Evaluate the results using phoneme comparison and MOS scores.

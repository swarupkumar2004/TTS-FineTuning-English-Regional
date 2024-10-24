import coqui_tts
from coqui_tts.utils.text.symbols import phonemes
from coqui_tts.trainer import Trainer
from coqui_tts.configs import TTSConfig
from datasets import load_dataset

def load_data(filepath):
    with open(filepath, "r") as f:
        data = f.readlines()
    return data

def fine_tune_english_tts():
    dataset = load_data("../datasets/english_technical_dataset.txt")

    # Load pre-trained Coqui TTS model
    config = TTSConfig()
    model = coqui_tts.load_model(config, pretrained_model="en")

    # Prepare training
    trainer = Trainer(config)
    trainer.prepare_dataset(dataset, phoneme_cache=phonemes)

    # Fine-tune model
    trainer.train(batch_size=16, learning_rate=0.001, epochs=20)

    # Save the fine-tuned model
    model.save("../models/english_tts_model.pth")

if __name__ == "__main__":
    fine_tune_english_tts()

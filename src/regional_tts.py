import coqui_tts
from coqui_tts.trainer import Trainer
from coqui_tts.configs import TTSConfig
from datasets import load_dataset

def load_data(filepath):
    with open(filepath, "r") as f:
        data = f.readlines()
    return data

def fine_tune_regional_tts():
    dataset = load_data("../datasets/regional_language_dataset.txt")

    # Load pre-trained model for regional language (Coqui TTS or SpeechT5)
    config = TTSConfig(language="regional_language")
    model = coqui_tts.load_model(config, pretrained_model="regional_language_model")

    # Prepare training
    trainer = Trainer(config)
    trainer.prepare_dataset(dataset)

    # Fine-tune model
    trainer.train(batch_size=16, learning_rate=0.001, epochs=20)

    # Save the fine-tuned model
    model.save("../models/regional_tts_model.pth")

if __name__ == "__main__":
    fine_tune_regional_tts()

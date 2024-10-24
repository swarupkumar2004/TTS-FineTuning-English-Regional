import torch
from torch.quantization import quantize_dynamic
from coqui_tts import TTS

def optimize_model_for_inference(model_path):
    model = TTS.load(model_path)
    
    # Quantization for fast inference
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    # Save the quantized model
    torch.save(quantized_model.state_dict(), model_path.replace(".pth", "_quantized.pth"))
    
    return quantized_model

def benchmark_inference_speed(model, test_input):
    import time
    start_time = time.time()
    output = model.synthesize(test_input)
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    model_path = "../models/english_tts_model.pth"
    quantized_model = optimize_model_for_inference(model_path)
    test_input = "API is a crucial part of modern web development."
    inference_time = benchmark_inference_speed(quantized_model, test_input)
    print(f"Inference Time: {inference_time} seconds")

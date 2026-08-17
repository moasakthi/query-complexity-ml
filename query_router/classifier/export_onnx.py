"""Export + quantize the fine-tuned classifier — DESIGN.md §6.4."""
import argparse
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export(model_dir: str, output_path: str, quantize: bool = True, max_length: int = 64):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy = tokenizer(
        "example query", return_tensors="pt", padding="max_length", max_length=max_length
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,  # the dynamo-based exporter mis-infers shapes for this architecture as of torch 2.12
    )
    tokenizer.save_pretrained(os.path.dirname(output_path))
    print(f"Exported ONNX model to {output_path}")

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = output_path.replace(".onnx", ".int8.onnx")
        quantize_dynamic(output_path, quantized_path, weight_type=QuantType.QInt8)
        print(f"Quantized INT8 model written to {quantized_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/classifier")
    parser.add_argument("--output", default="models/classifier_onnx/model.onnx")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    export(args.model_dir, args.output, quantize=not args.no_quantize)


if __name__ == "__main__":
    main()

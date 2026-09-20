from transformers import AutoModelForSequenceClassification

model_path = "model_indobert_final\\model_indobert_final"
model, info = AutoModelForSequenceClassification.from_pretrained(
    model_path, output_loading_info=True
)

print("Missing keys:   ", info.get("missing_keys"))
print("Unexpected keys:", info.get("unexpected_keys"))
print("Mismatched keys:", info.get("mismatched_keys"))

w = model.classifier.weight.detach().cpu()
print("\nclassifier.weight std :", float(w.std()))
print("classifier.bias       :", model.classifier.bias.detach().cpu().tolist())
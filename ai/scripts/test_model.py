from pathlib import Path
import onnxruntime as ort


MODEL_PATH = Path(
    "ai/models/object_detection/yolov8n.onnx"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

print("[INFO] Loading model...")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

print("\n========== MODEL ==========")
print(MODEL_PATH)

print("\n========== INPUTS ==========")

for inp in session.get_inputs():
    print("Name :", inp.name)
    print("Shape:", inp.shape)
    print("Type :", inp.type)

print("\n========== OUTPUTS ==========")

for output in session.get_outputs():
    print("Name :", output.name)
    print("Shape:", output.shape)
    print("Type :", output.type)

print("\n========== PROVIDERS ==========")

for provider in session.get_providers():
    print("-", provider)

print("\n[OK] Model loaded successfully.")
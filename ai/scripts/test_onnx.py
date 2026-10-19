import onnxruntime as ort


print("ONNX Runtime version:", ort.__version__)

providers = ort.get_available_providers()

print("Available providers:")
for provider in providers:
    print(" -", provider)

if "CPUExecutionProvider" in providers:
    print("\nCPU inference is available.")
else:
    print("\nWARNING: CPUExecutionProvider not found.")
# tflite_to_c.py
import sys
fn = "./model/model.tflite"
out = "../05_inference_gyro/model.h"
data = open(fn,"rb").read()
arr = ",".join(str(b) for b in data)
with open(out,"w") as f:
    f.write(f"// Automatically generated from {fn}\n")
    f.write(f"const unsigned char model[] = {{\n")
    # break into lines
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        f.write("  " + ",".join(str(b) for b in chunk) + ",\n")
    f.write("};\n")
    f.write(f"const unsigned int model_len = {len(data)};\n")
print("Wrote", out)

# train_mlp.py
"""
Usage:
    python train_mlp.py

Requirements:
    pip install numpy pandas scikit-learn tensorflow==2.12.0
(you can use another TF2.x but ensure tflite converter api is available)
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ---------- USER CONFIG ----------
CSV_GLOBS = ["up.csv", "down.csv", "left.csv", "right.csv"]  # adjust filenames if needed
TIMESTEPS = 200          # target samples per repetition (2s @ 100Hz)
CHANNELS = 3             # gx,gy,gz
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 40
DENSE_UNITS = 64        # < 100 as requested
MODEL_TFLITE = "model.tflite"
MODEL_TFLITE_QUANT = "model_quant.tflite"
# ---------------------------------

def load_all(csv_globs):
    X = []
    y = []
    classes = []
    for csvname in csv_globs:
        if not os.path.exists(csvname):
            print(f"Warning: {csvname} not found, skipping.")
            continue
        # infer gesture label from file name if column 'gesture' not present
        df = pd.read_csv(csvname)
        # ensure columns exist
        if not {'gesture','rep','t_ms','gx','gy','gz'}.issubset(df.columns):
            raise RuntimeError(f"{csvname} missing required columns.")
        # group by repetition
        for (gesture, rep), group in df.groupby(['gesture','rep']):
            # sort by time
            group = group.sort_values('t_ms')
            gx = group['gx'].to_numpy()
            gy = group['gy'].to_numpy()
            gz = group['gz'].to_numpy()
            arr = np.vstack([gx, gy, gz]).T  # shape (N,3)
            X.append(arr)
            y.append(gesture)
            if gesture not in classes:
                classes.append(gesture)
    return X, y, classes

def pad_or_trim(arr, target_len):
    """Resample/pad/trim a (N,3) array to shape (target_len,3)"""
    n = arr.shape[0]
    if n == target_len:
        return arr
    if n < target_len:
        # pad with last row
        pad_count = target_len - n
        pad_rows = np.repeat(arr[-1:, :], pad_count, axis=0)
        return np.vstack([arr, pad_rows])
    else:
        # trim (or we could downsample) — take first target_len samples
        return arr[:target_len, :]

def build_dataset(X_list, y_list, classes, target_len):
    # map labels to ints
    classes = sorted(classes)
    cls_to_idx = {c:i for i,c in enumerate(classes)}
    N = len(X_list)
    X = np.zeros((N, target_len, CHANNELS), dtype=np.float32)
    Y = np.zeros((N,), dtype=np.int32)
    for i, arr in enumerate(X_list):
        arr2 = pad_or_trim(arr, target_len)
        X[i] = arr2
        Y[i] = cls_to_idx[y_list[i]]
    return X, Y, classes

def normalize_train(X_train, X_val):
    # compute mean/std on train across (samples, timesteps)
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    return X_train_n, X_val_n, mean, std

def make_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(DENSE_UNITS, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def representative_gen(X_train):
    # generator yields batches for quantization calibration (must yield float32 arrays)
    def gen():
        for i in range(min(100, X_train.shape[0])):
            sample = X_train[i:i+1].astype(np.float32)
            yield [sample]
    return gen

def convert_to_tflite(keras_model, X_train):
    # save standard float tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(MODEL_TFLITE, "wb").write(tflite_model)
    print(f"Saved float TFLite model to {MODEL_TFLITE}")

    # quantize to int8 (full-integer) using representative dataset
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant = converter.convert()
    open(MODEL_TFLITE_QUANT, "wb").write(tflite_quant)
    print(f"Saved INT8 quantized TFLite model to {MODEL_TFLITE_QUANT}")

def main():
    print("Loading CSVs...")
    X_list, y_list, classes = load_all(CSV_GLOBS)
    if len(X_list) == 0:
        print("No data found. Make sure your CSV files are here and named properly.")
        return
    print(f"Found {len(X_list)} trials for classes: {sorted(classes)}")

    X, Y, classes = build_dataset(X_list, y_list, classes, TIMESTEPS)
    print(f"Built dataset: X={X.shape}, Y={Y.shape}")

    # split
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y, random_state=RANDOM_STATE)
    print(f"Train/val split: {X_train.shape[0]} / {X_val.shape[0]}")

    # normalize
    X_train_n, X_val_n, mean, std = normalize_train(X_train, X_val)

    # model
    model = make_model((TIMESTEPS, CHANNELS), num_classes=len(classes))
    model.summary()

    # callbacks
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)]

    # train
    history = model.fit(X_train_n, y_train,
                        validation_data=(X_val_n, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks)

    # evaluate
    val_loss, val_acc = model.evaluate(X_val_n, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    # convert to tflite (float + quant)
    convert_to_tflite(model, X_train_n)

    # save label mapping
    mapping = {"classes": classes, "mean": mean.tolist(), "std": std.tolist(), "timesteps": TIMESTEPS}
    import json
    open("label_map.json","w").write(json.dumps(mapping))
    print("Saved label_map.json (classes, mean/std)")

if __name__ == "__main__":
    main()

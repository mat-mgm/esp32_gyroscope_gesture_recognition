# train_mlp.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json

# ---------- CONFIG ----------
CSV_GLOBS = ["../02_capture_data/data/up.csv", "../02_capture_data/data/down.csv", "../02_capture_data/data/left.csv", "../02_capture_data/data/right.csv", "../02_capture_data/data/front.csv", "../02_capture_data/data/still.csv"]
TIMESTEPS = 200
CHANNELS = 6      # gx,gy,gz,ax,ay,az
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50
DENSE_UNITS = 128
MODEL_DIR = "./model"
MODEL_TFLITE = os.path.join(MODEL_DIR, "model.tflite")
MODEL_TFLITE_QUANT = os.path.join(MODEL_DIR, "model_quant.tflite")
# ----------------------------

def load_all(csv_globs):
    X, y, classes = [], [], []
    for csvname in csv_globs:
        if not os.path.exists(csvname):
            print(f"Warning: {csvname} not found, skipping.")
            continue
        df = pd.read_csv(csvname)
        if not {'gesture','rep','t_ms','gx','gy','gz','ax','ay','az'}.issubset(df.columns):
            raise RuntimeError(f"{csvname} missing required columns.")
        # each gesture repetition becomes one sample
        for (gesture, rep), group in df.groupby(['gesture','rep']):
            group = group.sort_values('t_ms')
            arr = group[['gx','gy','gz','ax','ay','az']].to_numpy()
            X.append(arr)
            y.append(gesture)
            if gesture not in classes:
                classes.append(gesture)
    return X, y, classes

def pad_or_trim(arr, target_len):
    n = arr.shape[0]
    if n == target_len:
        return arr
    if n < target_len:
        pad_rows = np.zeros((target_len - n, arr.shape[1]), dtype=arr.dtype)
        return np.vstack([arr, pad_rows])
    else:
        return arr[:target_len, :]

def clean_array(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -50, 50)  # clip extreme gyro/acc spikes
    return arr

def build_dataset(X_list, y_list, classes, target_len):
    classes = sorted(classes)
    cls_to_idx = {c:i for i,c in enumerate(classes)}
    N = len(X_list)
    X = np.zeros((N, target_len, CHANNELS), dtype=np.float32)
    Y = np.zeros((N,), dtype=np.int32)
    for i, arr in enumerate(X_list):
        arr = clean_array(arr)
        X[i] = pad_or_trim(arr, target_len)
        Y[i] = cls_to_idx[y_list[i]]
    return X, Y, classes

""" def normalize_train(X_train, X_val):
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
    return (X_train - mean)/std, (X_val - mean)/std, mean, std """

def normalize_train(X_train, X_val):
    mean = np.nanmean(X_train, axis=(0,1), keepdims=True)
    std = np.nanstd(X_train, axis=(0,1), keepdims=True)
    std[std < 1e-6] = 1.0   # avoid division by zero
    return (X_train - mean)/std, (X_val - mean)/std, mean, std

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
    def gen():
        for i in range(min(100, X_train.shape[0])):
            yield [X_train[i:i+1].astype(np.float32)]
    return gen

def convert_to_tflite(model, X_train):
    os.makedirs(MODEL_DIR, exist_ok=True)
    # float32 TFLite
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    open(MODEL_TFLITE, "wb").write(tflite_model)
    print(f"Saved float TFLite to {MODEL_TFLITE}")
    # quantized int8
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant = converter.convert()
    open(MODEL_TFLITE_QUANT, "wb").write(tflite_quant)
    print(f"Saved INT8 quantized TFLite to {MODEL_TFLITE_QUANT}")

def main():
    X_list, y_list, classes = load_all(CSV_GLOBS)
    if len(X_list) == 0:
        print("No data found. Aborting.")
        return
    X, Y, classes = build_dataset(X_list, y_list, classes, TIMESTEPS)
    
    # Debug check
    print("Any NaN in X?", np.isnan(X).any())
    print("Any Inf in X?", np.isinf(X).any())
    print("Dataset shape:", X.shape)
    print("Label counts:", dict(zip(*np.unique(Y, return_counts=True))))

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y, random_state=RANDOM_STATE)

    X_train_n, X_val_n, mean, std = normalize_train(X_train, X_val)

    model = make_model((TIMESTEPS, CHANNELS), len(classes))
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)]
    history = model.fit(X_train_n, y_train,
                        validation_data=(X_val_n, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks)
    val_loss, val_acc = model.evaluate(X_val_n, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    convert_to_tflite(model, X_train_n)
    
    # save label mapping
    mapping = {"classes": classes, "mean": mean.tolist(), "std": std.tolist(), "timesteps": TIMESTEPS}
    open(os.path.join(MODEL_DIR, "label_map.json"), "w").write(json.dumps(mapping))
    print("Saved label_map.json")
    
if __name__ == "__main__":
    main()


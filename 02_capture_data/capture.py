# capture.py
import serial, time, os, sys

PORT = "COM3"  # adjust to your port
BAUD = 115200
DUR = 2000     # ms; length of each recording
DATA_DIR = "./data"

def capture_one(ser, duration_ms=DUR):
    ser.reset_input_buffer()
    ser.write(f"REC,{duration_ms}\n".encode())
    started = False
    lines = []
    t0 = time.time()
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            if time.time() - t0 > 5 and not started:
                raise RuntimeError("No START received")
            continue
        if line == "START":
            started = True
            continue
        if line == "END":
            break
        if started:
            lines.append(line)
    return lines

def get_last_rep(filename):
    if not os.path.exists(filename):
        return 0
    last_rep = 0
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("gesture"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    rep_idx = int(parts[1])
                    if rep_idx > last_rep:
                        last_rep = rep_idx
                except:
                    pass
    return last_rep

def batch_capture(label, reps, duration_ms=DUR):
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(DATA_DIR, f"{label}.csv")
    file_exists = os.path.exists(filename)
    start_rep = get_last_rep(filename)

    with open(filename, "a") as f:
        if not file_exists:
            f.write("gesture,rep,t_ms,gx,gy,gz,ax,ay,az\n")
        for r in range(1, reps + 1):
            rep_idx = start_rep + r
            input(f"\nPress ENTER for repetition {rep_idx} of '{label}'...")
            print("Recording...")
            lines = capture_one(ser, duration_ms)
            for line in lines:
                f.write(f"{label},{rep_idx},{line}\n")
            print(f"Saved repetition {rep_idx} ({len(lines)} samples)")

    ser.close()
    print(f"\nAll {reps} repetitions saved to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python capture.py <label> <reps> [port]")
    else:
        if len(sys.argv) >= 4:
            PORT = sys.argv[3]
        batch_capture(sys.argv[1], int(sys.argv[2]))

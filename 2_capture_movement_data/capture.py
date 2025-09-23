# capture.py
import serial, time, sys, os

PORT = "COM3"            # change to your port (e.g. /dev/ttyUSB0)
BAUD = 115200
DUR = 2000               # ms; length of each recording

def capture_one(ser, duration_ms=1000):
    """Capture one repetition from Arduino and return list of CSV lines (no header)."""
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
    """Return last repetition index in existing CSV, or 0 if new file."""
    if not os.path.exists(filename):
        return 0
    last_rep = 0
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("gesture"):  # skip header
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    rep_idx = int(parts[1])
                    if rep_idx > last_rep:
                        last_rep = rep_idx
                except ValueError:
                    pass
    return last_rep

def batch_capture(label, reps, duration_ms=1000):
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(1)

    filename = f"{label}.csv"
    file_exists = os.path.exists(filename)

    start_rep = get_last_rep(filename)

    with open(filename, "a") as f:
        if not file_exists:
            f.write("gesture,rep,t_ms,gx,gy,gz\n")

        for r in range(1, reps + 1):
            rep_idx = start_rep + r
            input(f"\nPress ENTER when ready for repetition {rep_idx}... ")
            print("Recording...")

            lines = capture_one(ser, duration_ms)
            for line in lines:
                f.write(f"{label},{rep_idx},{line}\n")

            print(f"Saved repetition {rep_idx} with {len(lines)} samples")

    ser.close()
    print(f"\nAll {reps} new repetitions appended to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python capture.py <label> <reps> [port]")
    else:
        if len(sys.argv) >= 4: PORT = sys.argv[3]
        batch_capture(sys.argv[1], int(sys.argv[2]))

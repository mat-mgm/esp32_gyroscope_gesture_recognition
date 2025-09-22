# capture.py
import serial, time, sys, os

PORT = "COM3"            # change to your port (e.g. /dev/ttyUSB0)
BAUD = 115200
DUR = 1000               # ms; length of each recording

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

def batch_capture(label, reps, duration_ms=1000):
    """Capture N repetitions and save into one CSV file."""
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(1)

    filename = f"{label}.csv"
    with open(filename, "w") as f:
        f.write("rep,t_ms,gx,gy,gz\n")  # add repetition index as first column

        for r in range(1, reps + 1):
            input(f"\nPress ENTER when ready for repetition {r}/{reps}...")
            print("Recording...")

            lines = capture_one(ser, duration_ms)
            # prepend repetition index to each line
            for line in lines:
                f.write(f"{r},{line}\n")

            print(f"Saved repetition {r} with {len(lines)} samples")

    ser.close()
    print(f"\nAll {reps} repetitions saved to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python capture.py <label> <reps> [port]")
    else:
        if len(sys.argv) >= 4: PORT = sys.argv[3]
        batch_capture(sys.argv[1], int(sys.argv[2]))

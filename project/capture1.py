# capture.py
import serial, time, sys, os

PORT = "COM3"            # change to your port (e.g. /dev/ttyUSB0)
BAUD = 115200
DUR = 1000               # ms; used only when sending "REC,ms" command

def capture(label, rep, filename=None):
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(1)
    ser.reset_input_buffer()
    ser.write(f"REC,{DUR}\n".encode())
    started = False
    lines = []
    t0 = time.time()
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if not line: 
            # allow small timeout
            if time.time() - t0 > 5 and not started:
                raise RuntimeError("No START received")
            continue
        if line == "START":
            started = True
            continue
        if line == "END":
            break
        if started:
            # line: t_ms,gx,gy,gz
            lines.append(line)
    ser.close()
    if filename is None:
        filename = f"{label}_rep{rep:02d}.csv"
    with open(filename, "w") as f:
        f.write("t_ms,gx,gy,gz\n")
        f.write("\n".join(lines))
    print("Saved", filename)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python capture.py <label> <rep> [port]")
    else:
        if len(sys.argv) >= 4: PORT = sys.argv[3]
        capture(sys.argv[1], int(sys.argv[2]))

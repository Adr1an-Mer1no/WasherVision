
import time
import serial


class GRBLController:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(
            port,
            baudrate,
            timeout=1,
            write_timeout=2
        )

        # Opening serial may reset the Arduino Uno.
        time.sleep(2)
        self.ser.reset_input_buffer()

    def send_command(self, command, timeout=5):
        """Send a non-motion GRBL command."""
        allowed = {"$$", "$G", "$I", "$#", "$N"}
        if command not in allowed:
            raise ValueError("Command not allowed in diagnostic mode")

        self.ser.write((command + "\n").encode("ascii"))
        self.ser.flush()

        deadline = time.monotonic() + timeout
        lines = []

        while time.monotonic() < deadline:
            line = self.ser.readline().decode(
                "ascii", errors="replace"
            ).strip()

            if not line:
                continue

            lines.append(line)

            if line == "ok":
                return lines

            if line.startswith("error:"):
                raise RuntimeError(line)

        raise TimeoutError("No complete GRBL response")

    def get_status(self, timeout=3):
        """Read a real-time GRBL status report."""
        self.ser.reset_input_buffer()
        self.ser.write(b"?")
        self.ser.flush()

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            line = self.ser.readline().decode(
                "ascii", errors="replace"
            ).strip()

            if line.startswith("<") and line.endswith(">"):
                return line

        raise TimeoutError("No GRBL status report")

    def close(self):
        self.ser.close()


if __name__ == "__main__":
    port = input("Arduino serial port: ").strip()
    cnc = None

    try:
        cnc = GRBLController(port)

        print("\nGRBL information:")
        for line in cnc.send_command("$I"):
            print(line)

        print("\nMachine status:")
        print(cnc.get_status())

        print("\nGRBL connection successful.")

    finally:
        if cnc is not None:
            cnc.close()


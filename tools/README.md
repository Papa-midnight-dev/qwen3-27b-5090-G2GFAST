# Companion Linux Tools

Standalone utilities bundled with the `moonshot` fork of `mkyt/qwen3-27b-5090`.

## 1. `rtx5090-undervolt.py` — Linux RTX 5090 True NVML Undervolt

Performs an MSI-Afterburner-style **true V/F curve undervolt** on Linux using `libnvidia-ml.so.1` (no X11/Coolbits required):
1. Shifts the P0 graphics V/F curve **upward** (`--core-offset`, e.g. `+180 MHz`) so every clock point runs at lower voltage.
2. Flattens the top of the curve with a boost ceiling (`--max-clock`, e.g. `2750–2800 MHz`) while keeping `--min-clock 210` so idle stays at P8 (~24 W).
3. Sets a longevity power limit (`--watts 450`, **hard-capped at `<= 550 W`** in code).
4. Applies a conservative GDDR7 memory offset (`--mem-offset 1000` to `1500`) to boost memory-bound LLM decode.

```bash
# Read-only status (no sudo required):
python3 tools/rtx5090-undervolt.py --status

# Apply 450W conservative profile (+180 MHz V/F, 2750 MHz cap, +1000 MHz GDDR7):
sudo python3 tools/rtx5090-undervolt.py --apply

# Apply tuned 450W profile (+180 MHz V/F, 2800 MHz cap, +1500 MHz GDDR7):
sudo python3 tools/rtx5090-undervolt.py --apply --watts 450 --core-offset 180 --max-clock 2800 --mem-offset 1500

# Restore factory defaults anytime:
sudo python3 tools/rtx5090-undervolt.py --reset
```

To persist across reboots with systemd:
```bash
sudo install -m 0755 tools/rtx5090-undervolt.py /usr/local/bin/rtx5090-undervolt.py
sudo install -m 0644 tools/rtx5090-undervolt.service /etc/systemd/system/rtx5090-undervolt.service
sudo systemctl daemon-reload && sudo systemctl enable --now rtx5090-undervolt.service
```
Override defaults persistently via `/etc/rtx5090-undervolt.env` (`UV_WATTS`, `UV_CORE_OFFSET`, `UV_MAX_CLOCK`, `UV_MEM_OFFSET`).

*Note: silicon quality varies — always verify greedy SHA stability on your own card before persisting aggressive offsets.*

## 2. `q27-fwd.c` — Zero-Fork C `epoll` TCP↔UDS Forwarder

Allows running the `q27-engine` container with `--network none` (zero outbound network access) over a bind-mounted Unix domain socket (`LISTEN_UNIX=/run/q27/api.sock`) while serving local OpenAI/Anthropic clients on `127.0.0.1:8000` with sub-0.15 ms overhead.

```bash
gcc -O2 -Wall -Wextra tools/q27-fwd.c -o q27-fwd
./q27-fwd 127.0.0.1:8000 unix:/run/q27-engine/api.sock
```

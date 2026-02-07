# Google Minesweeper Bot

A Python bot that automatically plays [Google Minesweeper](https://www.google.com/fbx?fbx=minesweeper) using screen capture and automated clicking.

## How It Works

The bot captures your screen, detects the minesweeper grid using computer vision (OpenCV), reads cell states via color sampling, solves using constraint satisfaction with backtracking enumeration, and clicks/flags cells using PyAutoGUI.

## Quick Start

### macOS / Linux

1. Open [Google Minesweeper](https://www.google.com/fbx?fbx=minesweeper) in your browser
2. Make sure the game board is fully visible on screen
3. Run the launcher:

```bash
./run.sh
```

On first run this creates a virtual environment and installs all dependencies automatically. You just need [Python 3.8+](https://www.python.org/downloads/) installed.

**macOS permissions:** Your terminal app needs **Screen Recording** and **Accessibility** access (System Settings > Privacy & Security). macOS will prompt you on first run.

### Windows

1. Open [Google Minesweeper](https://www.google.com/fbx?fbx=minesweeper) in your browser
2. Make sure the game board is fully visible on screen
3. Double-click **`run.bat`**, or run it from a terminal:

```
run.bat
```

On first run this creates a virtual environment and installs all dependencies automatically. You just need [Python 3.8+](https://www.python.org/downloads/) installed. No special permissions are required; DPI scaling is auto-detected.

### Manual Setup

If you prefer to manage dependencies yourself:

```bash
pip install pyautogui opencv-python numpy pillow
python bot.py
```

The bot will wait 3 seconds before starting, giving you time to switch to the browser.

## Stopping the Bot

- **Press Esc** at any time to stop the bot. This works even when the browser is focused.
- **Move your mouse to the top-left corner** of the screen to trigger PyAutoGUI's failsafe.

## Options

Pass flags after the launcher script or directly to `bot.py`:

```bash
./run.sh --debug        # Save annotated screenshots to debug/
./run.sh --step         # Step mode: press SPACE before each click, 'q' to quit
./run.sh --delay 0.05   # Set per-click delay in seconds (default: 0.01s)
```

### Supported Difficulties

| Difficulty | Grid   | Mines |
|------------|--------|-------|
| Easy       | 10x8   | 10    |
| Medium     | 18x14  | 40    |
| Hard       | 24x20  | 99    |

The bot auto-detects the grid size from the screenshot.

## How the Solver Works

The solver uses a multi-pass approach, from fast heuristics to full enumeration:

1. **Endgame check** -- if all remaining unknowns are mines (or no mines remain)
2-4. **Iterative inference chaining** -- loops basic flagging, basic safe cells, and subset analysis until convergence. Allows inferences to chain across the entire board (e.g., finding mines on one side enables finding safe cells on the other side in the same iteration)
5. **Global border enumeration** -- only runs if no deterministic moves found. Uses backtracking with constraint-based ordering (most-constrained cells first) to find moves or calculate mine probabilities
6. **Probability-based guess** -- pick the cell with lowest mine probability
7. **Random fallback** -- prefer corners and edges

## Project Structure

```
bot.py      Main game loop and CLI
screen.py   Screenshot capture and coordinate translation
grid.py     Grid detection (OpenCV) and cell state reading
solver.py   Multi-pass constraint satisfaction solver
```

## Debug Mode

Run with `--debug` to save annotated screenshots to `debug/`:

- `debug/move_NNNN.png` -- screenshots with grid overlay and move highlights (green = click, red = flag)
- `debug/cell_samples.txt` -- per-cell color sampling details

## Limitations

- The browser window with the game must be fully visible (not overlapped)
- Very large borders (>70 unknown cells adjacent to numbers) fall back to region-based solving

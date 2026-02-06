# Google Minesweeper Bot

A Python bot that automatically plays [Google Minesweeper](https://www.google.com/fbx?fbx=minesweeper) using screen capture and automated clicking.

## How It Works

The bot captures your screen, detects the minesweeper grid using computer vision (OpenCV), reads cell states via color sampling, solves using constraint satisfaction with backtracking enumeration, and clicks/flags cells using PyAutoGUI.

## Requirements

- Python 3.8+
- macOS or Windows
- Google Minesweeper open in a browser

## Setup

```bash
pip install pyautogui opencv-python numpy pillow
```

On macOS, you'll need to grant screen recording and accessibility permissions to your terminal app (System Settings > Privacy & Security). Windows requires no special permissions; DPI scaling is auto-detected.

## Usage

1. Open [Google Minesweeper](https://www.google.com/fbx?fbx=minesweeper) in your browser
2. Make sure the game board is fully visible on screen
3. Run the bot:

```bash
python bot.py
```

The bot will wait 3 seconds before starting, giving you time to switch to the browser.

### Options

```bash
python bot.py --debug        # Save annotated screenshots to debug/
python bot.py --step         # Step mode: press SPACE before each click, 'q' to quit
python bot.py --delay 0.05   # Set per-click delay in seconds (default: 0.01s)
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
2. **Basic flagging** -- a numbered cell's unknown neighbors exactly match its remaining mine count
3. **Basic safe cells** -- a numbered cell already has all its mines flagged
4. **Subset analysis** -- deduce mines/safe cells from overlapping constraint pairs
5. **Global border enumeration** -- backtracking over all border cells (up to 40) to find deterministic moves or calculate mine probabilities
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
- Very large borders (>40 unknown cells adjacent to numbers) fall back to region-based solving

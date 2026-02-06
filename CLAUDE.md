# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python bot that plays Google Minesweeper using screen capture (PyAutoGUI) and automated clicking. Fully implemented and functional.

## Dependencies

```
pip install pyautogui opencv-python numpy pillow
```

## Running

```bash
python bot.py                    # Normal mode
python bot.py --debug            # Save debug screenshots + grid state to debug/
python bot.py --step             # Step mode: press SPACE before each click, 'q' to quit
python bot.py --delay 0.05       # Set per-click delay (default: 0.01s)
python bot.py --debug --step     # Both debug and step mode
```

## Architecture

Three modules + main bot loop:

1. **screen.py** - Screenshot capture and coordinate system translation (logical ↔ physical pixels)
2. **grid.py** - Grid detection via OpenCV connected components on RGB color masks, cell state reading via color sampling
3. **solver.py** - Multi-pass minesweeper solver with global border enumeration
4. **bot.py** - Main game loop, CLI args, debug image output

Main loop: Capture screenshot → Update grid → Solve → Execute moves → Wait 0.5s → Repeat

### Bot behavior notes
- First iteration double-clicks cells to start the game
- Within each iteration, flag moves chain without recapturing (flags update `grid.cells` directly); only click moves trigger a new screenshot
- If solver returns >40 cells, retries up to 3 times (board likely unreadable from win/loss overlay), then exits
- 0.5s post-move delay lets animation particles settle before next screenshot
- Flags are tracked internally via a `set` and also written to `grid.cells` immediately (not re-detected from screenshots)
- Before committing to a guess (iteration > 1), recaptures board with 0.5s delay in case cascading reveals haven't settled
- Always pauses for user confirmation (space/q) before executing a guess move
- Grid obscured detection: if >25% of cells have unrecognized backgrounds, assumes game is over
- Bot exits when solver has no moves left or grid is obscured

## Critical: Coordinate Systems

**This is the #1 source of bugs.** Two coordinate systems exist:

- **Logical pixels**: What `pyautogui.click()` expects. All coordinates stored as logical.
- **Physical pixels**: What `screenshot.getpixel()` uses (2x on Retina displays).

Conversion happens in `Screen.get_pixel()` and `Screen.logical_to_physical()`:
```python
physical = int(logical * scale)
```

## Grid Detection & Cell States

Grid detected by color-masking green/tan pixels, finding the largest connected component via OpenCV, then detecting cell size from checkerboard luminance transitions.

Cell states: `'unknown'` (green/unclicked), `'flag'`, `0-8` (revealed numbers)

### Color detection approach
- **Background sampling**: Sample 4 corners of each cell (offset = `max(8, cell_size//3)`) to determine green vs tan background, avoiding center icons/numbers
- **Tan-first priority**: If ANY corner is tan → revealed cell → detect number from center samples (checked first so corner samples landing on a neighbor's green don't misclassify revealed cells)
- **Green background**: If ANY corner is green → `'unknown'`
- **Flag tracking**: Flags tracked via internal `Grid.flagged_cells` set rather than screenshot color detection (avoids flag/3 confusion since both are red)
- **Number detection**: Sample 6x6 grid around center, match against known number colors with tolerance 55

### Known color pitfalls
- Flag color `(215, 75, 40)` and number 3 color `(211, 47, 47)` are very close — internal flag tracking solves this
- Cursor hover changes cell colors — 0.5s post-move delay handles this
- Click animation particles can pollute screenshots — same delay handles this

## Grid Configuration

Mine counts by `(rows, cols)` in `bot.py`:
```python
MINE_COUNTS = {
    (8, 10): 10,   # Easy
    (14, 18): 40,  # Medium
    (20, 24): 99,  # Hard
}
```

## Solver Strategy

Multi-pass approach in `solver.py`:

1. **Endgame check** — if mine_count known: all unknowns are mines, or no mines remain
2. **Basic flagging** — numbered cell has exactly N unknowns matching remaining mines
3. **Basic safe cells** — numbered cell has all mines flagged, unknowns are safe
4. **Subset/intersection analysis** — constraint pairs where one is subset of another
5. **Global border enumeration** — backtracking over all border cells together (≤40 cells), falls back to region-based enumeration for larger borders
6. **Probability-based guess** — pick lowest mine probability from enumeration results
7. **Random fallback** — prefer corners/edges, cells not adjacent to numbers

Key constants: `MAX_REGION_SIZE = 50`, `MAX_SOLUTIONS = 100_000`

## Debug Output

With `--debug`:
- `debug/move_NNNN.png` — annotated screenshots with grid overlay and move highlights
- `debug/cell_samples.txt` — per-cell color sampling details (background, flag checks)

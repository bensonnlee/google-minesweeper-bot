"""Main bot loop for playing Minesweeper."""

import argparse
import os
import sys
import time

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty
from PIL import Image, ImageDraw

from screen import Screen
from grid import detect_grid, Grid
from solver import Solver

# Mine counts by grid dimensions (rows, cols)
MINE_COUNTS = {
    (8, 10): 10,   # Easy
    (14, 18): 40,  # Medium
    (20, 24): 99,  # Hard
}


def wait_for_space():
    """Wait for user to press space (or 'q' to quit)."""
    print("Press SPACE for next click ('q' to quit)...", end='', flush=True)
    if sys.platform == 'win32':
        while True:
            ch = msvcrt.getch().decode('utf-8', errors='ignore')
            if ch == ' ':
                print()
                return True
            elif ch == 'q':
                print()
                return False
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == ' ':
                    print()
                    return True
                elif ch == 'q':
                    print()
                    return False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)



def save_debug_image(screenshot: Image.Image, grid: Grid, move_action: str,
                     move_cells: list, iteration: int):
    """Save an annotated debug image."""
    img = screenshot.copy()
    draw = ImageDraw.Draw(img)
    scale = grid.screen.scale

    x, y = int(grid.x * scale), int(grid.y * scale)
    cell_size = int(grid.cell_size * scale)

    # Draw grid lines
    for row in range(grid.rows + 1):
        y_pos = y + row * cell_size
        draw.line([(x, y_pos), (x + grid.cols * cell_size, y_pos)], fill=(128, 128, 128), width=1)
    for col in range(grid.cols + 1):
        x_pos = x + col * cell_size
        draw.line([(x_pos, y), (x_pos, y + grid.rows * cell_size)], fill=(128, 128, 128), width=1)

    # Highlight move cells
    if move_cells:
        color = (0, 255, 0) if move_action == 'click' else (255, 0, 0)
        for row, col in move_cells:
            cx = x + col * cell_size + cell_size // 2
            cy = y + row * cell_size + cell_size // 2
            r = cell_size // 3
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=color, width=3)

    img.save(f"{grid.debug_dir}/move_{iteration:04d}.png")


CELL_DISPLAY = {'unknown': ' . ', 'flag': ' F ', 0: '   '}

def print_grid_state(grid: Grid):
    """Print current grid state to console."""
    print("\nCurrent grid state:")
    print("  " + " ".join(f"{c:2}" for c in range(grid.cols)))
    print("  " + "-" * (grid.cols * 3))
    for row in range(grid.rows):
        cells = [CELL_DISPLAY.get(grid.cells[row][col], f" {grid.cells[row][col]} ")
                 for col in range(grid.cols)]
        print(f"{row:2}|" + "".join(cells))
    print()


def _is_grid_obscured(grid):
    total_cells = grid.rows * grid.cols
    if grid.unrecognized_count > total_cells // 4:
        print(f"Grid obscured ({grid.unrecognized_count}/{total_cells} cells unrecognized) - game likely over.")
        return True
    return False


def _capture_and_solve(screen, grid, mine_count, debug):
    """Capture screenshot, update grid, and solve. Retries up to 3 times if
    the solver returns too many cells (board likely unreadable).

    Returns (action, cells, solver, screenshot), or (None, [], None, screenshot)
    if the grid is obscured or unreadable.
    """
    for attempt in range(3):
        screenshot = screen.capture()
        grid.update(screenshot)

        if _is_grid_obscured(grid):
            return None, [], None, screenshot

        if debug:
            print_grid_state(grid)

        solver = Solver(grid.cells, mine_count=mine_count)
        action, cells = solver.get_move()

        if action is None or len(cells) <= 40:
            return action, cells, solver, screenshot

        if attempt < 2:
            print(f"Move: {action} on {len(cells)} cells - too many, retrying...")
            time.sleep(0.5)

    print(f"Move: {action} on {len(cells)} cells - board unreadable, game likely over.")
    return None, [], None, screenshot


def _execute_moves(action, cells, solver, screenshot, grid, mine_count,
                   iteration, debug, step, delay):
    """Execute solver moves, chaining flag moves without recapturing.

    Returns False if the user requested quit, True otherwise.
    """
    move_num = 0
    while action is not None:
        move_num += 1
        if debug:
            print(f"[{move_num}] {action} on {len(cells)} cells {cells}")
        else:
            print(f"[{move_num}] {action} on {len(cells)} cells")

        if solver.is_guess and iteration > 1:
            if solver.guess_probability is not None:
                safe_pct = (1 - solver.guess_probability) * 100
                print(f"    No guaranteed move. Best guess: {safe_pct:.0f}% safe.")
            else:
                print(f"    No guaranteed move. Random guess.")
            if not wait_for_space():
                print("Quit requested.")
                return False

        if debug:
            save_debug_image(screenshot, grid, action, cells, iteration)

        for row, col in cells:
            if step and action == 'click':
                print(f"    Next: click ({row}, {col})")
                if not wait_for_space():
                    print("Quit requested.")
                    return False

            if action == 'click':
                grid.click_cell(row, col)
                if iteration == 1:
                    time.sleep(0.1)
                    grid.click_cell(row, col)
            elif action == 'flag':
                grid.flag_cell(row, col)
                grid.cells[row][col] = 'flag'
            time.sleep(delay)

        # After clicks, we need a new screenshot to see revealed cells
        if action == 'click':
            break

        # After flags, re-run solver on updated internal state (no screenshot needed)
        solver = Solver(grid.cells, mine_count=mine_count)
        action, cells = solver.get_move()
        if action is not None and len(cells) > 40:
            break  # Bail out, will recapture next iteration

    return True


def play(debug: bool = False, delay: float = 0.1, step: bool = False):
    """Main game loop.

    Args:
        debug: If True, save annotated screenshots and print state.
        delay: Delay between actions in seconds.
        step: If True, wait for space press before each cell click.
    """
    print("Minesweeper Bot starting...")
    print("Please ensure the Google Minesweeper game is visible on screen.")
    print("Starting in 3 seconds...")
    time.sleep(3)

    screen = Screen()
    print(f"Screen scale factor: {screen.scale}")

    debug_dir = None
    if debug:
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug images will be saved to: {debug_dir}/")

    print("Detecting grid...")
    screenshot = screen.capture()
    try:
        grid_info = detect_grid(screenshot, screen.scale)
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure the Minesweeper game is visible and start a new game.")
        return

    print(f"Detected grid: {grid_info.cols}x{grid_info.rows}, cell_size={grid_info.cell_size}")
    grid = Grid(screen, grid_info, debug_dir=debug_dir)
    mine_count = MINE_COUNTS.get((grid.rows, grid.cols))

    # Game loop
    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        action, cells, solver, screenshot = _capture_and_solve(
            screen, grid, mine_count, debug)

        # Before committing to a guess, recapture in case cascading reveals
        # from the last click haven't been fully picked up yet.
        if action is not None and solver.is_guess and iteration > 1:
            print("Guess needed - recapturing board state...")
            time.sleep(0.5)
            screenshot = screen.capture()
            grid.update(screenshot)
            if _is_grid_obscured(grid):
                action = None
            else:
                solver = Solver(grid.cells, mine_count=mine_count)
                action, cells = solver.get_move()

        if action is None:
            print("Game finished - no more moves available.")
            break

        if not _execute_moves(action, cells, solver, screenshot, grid,
                              mine_count, iteration, debug, step, delay):
            return

        # Wait for animation effects to finish before next screenshot
        time.sleep(0.5)

    print("\nBot finished.")


def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Minesweeper Bot')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (save screenshots, print grid state)')
    parser.add_argument('--delay', type=float, default=0.01,
                        help='Delay between actions in seconds (default: 0.01)')
    parser.add_argument('--step', action='store_true',
                        help='Step mode: wait for space press before each click')

    args = parser.parse_args()
    play(debug=args.debug, delay=args.delay, step=args.step)


if __name__ == '__main__':
    main()

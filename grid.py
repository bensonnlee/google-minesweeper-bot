"""Grid detection and cell state reading."""

import math
from collections import namedtuple
from PIL import Image
import numpy as np

from screen import Screen


# Grid information returned by detect_grid
GridInfo = namedtuple('GridInfo', ['bounds', 'cell_size', 'rows', 'cols'])


# Color constants for detection
COLORS = {
    # Unclicked cells (light/dark green pattern)
    'green_light': (170, 215, 81),
    'green_dark': (162, 209, 73),

    # Revealed empty cells (light/dark tan pattern)
    'tan_light': (229, 194, 159),
    'tan_dark': (215, 184, 153),

    # Flag (orange-red, actual Google Minesweeper color)
    'flag': (215, 75, 40),

    # Number colors (center of cells)
    'numbers': {
        1: (25, 118, 210),    # Blue
        2: (56, 142, 60),     # Green
        3: (211, 47, 47),     # Red
        4: (123, 31, 162),    # Purple
        5: (240, 149, 54),    # Orange
        6: (66, 149, 165),    # Cyan/teal
        7: (66, 66, 66),      # Dark gray
        8: (158, 158, 158),   # Light gray
    },

    # Mine (red background when game over)
    'mine': (255, 0, 0),
}


def color_distance(c1: tuple, c2: tuple) -> float:
    """Calculate Euclidean distance between two RGB colors.

    Args:
        c1: First RGB tuple.
        c2: Second RGB tuple.

    Returns:
        Distance as float.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def is_color_match(color: tuple, target: tuple, tolerance: float = 50) -> bool:
    """Check if color matches target within tolerance.

    Args:
        color: RGB tuple to check.
        target: Target RGB tuple.
        tolerance: Maximum Euclidean distance for match.

    Returns:
        True if colors match within tolerance.
    """
    return color_distance(color, target) <= tolerance


GREEN_COLORS = [COLORS['green_light'], COLORS['green_dark']]
TAN_COLORS = [COLORS['tan_light'], COLORS['tan_dark']]
GRID_COLORS = GREEN_COLORS + TAN_COLORS


def is_any_match(color: tuple, targets: list, tolerance: float = 50) -> bool:
    """Check if color matches any of the target colors within tolerance."""
    return any(is_color_match(color, t, tolerance) for t in targets)


def is_green(color: tuple, tolerance: float = 50) -> bool:
    """Check if color is one of the green unclicked cell colors."""
    return is_any_match(color, GREEN_COLORS, tolerance)


def is_tan(color: tuple, tolerance: float = 50) -> bool:
    """Check if color is one of the tan revealed cell colors."""
    return is_any_match(color, TAN_COLORS, tolerance)


def detect_grid(screenshot: Image.Image, scale: float) -> GridInfo:
    """Detect the minesweeper grid in a screenshot.

    Args:
        screenshot: PIL Image (physical pixels).
        scale: Scale factor (physical/logical).

    Returns:
        GridInfo with bounds, cell_size, rows, cols.
    """
    # Convert to numpy for faster processing (ensure RGB, not RGBA)
    img_array = np.array(screenshot.convert('RGB'))

    # Find pixels matching minesweeper colors (tight tolerance)
    grid_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)

    # Check for minesweeper-specific green and tan colors
    for target_color in GRID_COLORS:
        dist = np.sqrt(np.sum((img_array.astype(float) - target_color) ** 2, axis=2))
        grid_mask[dist < 30] = 255

    if not np.any(grid_mask):
        raise ValueError("Could not find minesweeper grid in screenshot")

    # Find contiguous regions using connected components
    import cv2
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grid_mask)

    # Find the largest component (excluding background label 0)
    if num_labels <= 1:
        raise ValueError("Could not find minesweeper grid in screenshot")

    # stats columns: x, y, width, height, area
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    physical_x = stats[largest_idx, cv2.CC_STAT_LEFT]
    physical_y = stats[largest_idx, cv2.CC_STAT_TOP]
    physical_w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    physical_h = stats[largest_idx, cv2.CC_STAT_HEIGHT]

    # Detect cell size by finding color transition pattern
    # Scan horizontally across the grid to find cell boundaries
    cell_size_physical = _detect_cell_size(img_array, physical_x, physical_y, physical_w, physical_h)

    # Convert to logical pixels
    logical_x = int(physical_x / scale)
    logical_y = int(physical_y / scale)
    logical_w = int(physical_w / scale)
    logical_h = int(physical_h / scale)
    cell_size = int(cell_size_physical / scale)

    # Calculate rows and cols
    cols = round(logical_w / cell_size)
    rows = round(logical_h / cell_size)

    # Recalculate cell size to fit exactly
    cell_size = logical_w // cols

    return GridInfo(
        bounds=(logical_x, logical_y, logical_w, logical_h),
        cell_size=cell_size,
        rows=rows,
        cols=cols
    )


def _detect_cell_size(img_array: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """Detect cell size by analyzing color pattern transitions.

    The grid has alternating light/dark cells in a checkerboard pattern.
    We detect the repeating pattern spacing.
    """
    # Sample a horizontal line in the middle of the grid
    sample_y = y + h // 2
    sample_row = img_array[sample_y, x:x+w]

    # Check if each pixel is "light" or "dark" based on luminance
    luminance = 0.299 * sample_row[:, 0] + 0.587 * sample_row[:, 1] + 0.114 * sample_row[:, 2]

    # Find transitions between light and dark
    threshold = (luminance.max() + luminance.min()) / 2
    is_light = luminance > threshold

    # Count transitions - each cell boundary causes a transition
    transitions = np.where(np.diff(is_light.astype(int)) != 0)[0]

    if len(transitions) < 2:
        # Fallback: try different sample lines
        for offset in [h // 4, h * 3 // 4]:
            sample_y = y + offset
            sample_row = img_array[sample_y, x:x+w]
            luminance = 0.299 * sample_row[:, 0] + 0.587 * sample_row[:, 1] + 0.114 * sample_row[:, 2]
            threshold = (luminance.max() + luminance.min()) / 2
            is_light = luminance > threshold
            transitions = np.where(np.diff(is_light.astype(int)) != 0)[0]
            if len(transitions) >= 2:
                break

    if len(transitions) < 2:
        # Last resort: estimate from common cell sizes
        # Common sizes: 45 (easy), 30 (medium), 25 (hard) - scaled by 2 for retina
        return 60  # Default fallback

    # Find most common spacing between transitions (cell width)
    spacings = np.diff(transitions)
    # Filter out very small spacings (noise)
    spacings = spacings[spacings > 10]

    if len(spacings) == 0:
        return 60

    # Use median for robustness
    cell_size = int(np.median(spacings))

    return cell_size


class Grid:
    """Manages the minesweeper grid state and interactions."""

    def __init__(self, screen: Screen, grid_info: GridInfo, debug_dir: str = None):
        """Initialize grid.

        Args:
            screen: Screen instance for capturing and clicking.
            grid_info: GridInfo with grid bounds and dimensions.
            debug_dir: If set, save color sampling debug info to this directory.
        """
        self.screen = screen
        self.info = grid_info
        self.rows = grid_info.rows
        self.cols = grid_info.cols
        self.cell_size = grid_info.cell_size
        self.x, self.y, self.w, self.h = grid_info.bounds
        self.debug_dir = debug_dir

        # 2D array of cell states
        # 'unknown' | 'flag' | 0-8
        self.cells = [[None for _ in range(self.cols)] for _ in range(self.rows)]

        # Track flagged cells internally (more reliable than color detection)
        self.flagged_cells = set()

        # Count of cells where background was neither green nor tan (overlay/obstruction)
        self.unrecognized_count = 0

    def get_cell_center(self, row: int, col: int) -> tuple:
        """Get the center coordinates of a cell in logical pixels.

        Args:
            row: Row index (0-based).
            col: Column index (0-based).

        Returns:
            Tuple (x, y) of cell center in logical pixels.
        """
        cx = self.x + col * self.cell_size + self.cell_size // 2
        cy = self.y + row * self.cell_size + self.cell_size // 2
        return (cx, cy)

    def update(self, screenshot: Image.Image):
        """Update all cell states from screenshot.

        Args:
            screenshot: PIL Image (physical pixels).
        """
        self.unrecognized_count = 0
        for row in range(self.rows):
            for col in range(self.cols):
                self.cells[row][col] = self._get_cell_state(screenshot, row, col)

    def _get_cell_state(self, screenshot: Image.Image, row: int, col: int):
        """Determine the state of a single cell.

        Args:
            screenshot: PIL Image (physical pixels).
            row: Row index.
            col: Column index.

        Returns:
            'unknown' | 'flag' | 0-8
        """
        # Check internal flag tracking first (more reliable than color detection)
        if (row, col) in self.flagged_cells:
            return 'flag'

        cx, cy = self.get_cell_center(row, col)

        # Sample BACKGROUND at corners/edges (away from center icon/number)
        # Use ~40% of cell size to hit background, not the icon
        offset = max(8, self.cell_size // 3)
        bg_colors = [self.screen.get_pixel(screenshot, cx + dx, cy + dy)
                     for dx, dy in [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]]

        # Check tan FIRST so that a revealed cell adjacent to green cells
        # (where a corner sample may land on the neighbor's green) still gets
        # detected as a number, not misclassified as unknown.
        if any(is_tan(c, 60) for c in bg_colors):
            return self._detect_number(screenshot, cx, cy)

        if any(is_green(c) for c in bg_colors):
            return 'unknown'

        # Fallback: background is neither green nor tan (possible overlay/game over)
        self.unrecognized_count += 1
        return 'unknown'

    def _detect_number(self, screenshot: Image.Image, cx: int, cy: int) -> int:
        """Detect what number (0-8) is displayed at a revealed cell.

        Args:
            screenshot: PIL Image.
            cx, cy: Center of cell in logical pixels.

        Returns:
            Number 0-8.
        """
        # Sample points in a small grid around center to find number pixels
        sample_points = []
        for dx in range(-5, 6, 2):
            for dy in range(-5, 6, 2):
                color = self.screen.get_pixel(screenshot, cx + dx, cy + dy)
                sample_points.append(color)

        # Check each number color
        for number, target_color in COLORS['numbers'].items():
            for color in sample_points:
                if is_color_match(color, target_color, 55):
                    return number

        # No number found - empty cell (0)
        return 0

    def click_cell(self, row: int, col: int):
        """Left-click a cell to reveal it.

        Args:
            row: Row index.
            col: Column index.
        """
        cx, cy = self.get_cell_center(row, col)
        self.screen.click(cx, cy, button='left')

    def flag_cell(self, row: int, col: int):
        """Right-click a cell to flag/unflag it.

        Args:
            row: Row index.
            col: Column index.
        """
        cx, cy = self.get_cell_center(row, col)
        self.screen.click(cx, cy, button='right')
        # Track flagged cell internally
        self.flagged_cells.add((row, col))

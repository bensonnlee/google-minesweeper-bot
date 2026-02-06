"""Screen capture and coordinate translation."""

import pyautogui
from PIL import Image

pyautogui.PAUSE = 0


class Screen:
    """Handles screenshot capture and coordinate system translation.

    Manages the conversion between logical pixels (what pyautogui expects)
    and physical pixels (what screenshots contain on Retina displays).
    """

    def __init__(self):
        """Initialize screen and auto-detect scale factor."""
        # Get logical screen size
        self.logical_width, self.logical_height = pyautogui.size()

        # Take a screenshot to detect physical size
        screenshot = pyautogui.screenshot()
        self.physical_width, self.physical_height = screenshot.size

        # Calculate scale factor (e.g. 2.0 on macOS Retina, 1.25/1.5/2.0 for Windows DPI scaling)
        self.scale = self.physical_width / self.logical_width

    def capture(self) -> Image.Image:
        """Capture a screenshot of the entire screen.

        Returns:
            PIL Image in physical pixels.
        """
        return pyautogui.screenshot()

    def get_pixel(self, screenshot: Image.Image, logical_x: int, logical_y: int) -> tuple:
        """Get pixel color at logical coordinates.

        Args:
            screenshot: PIL Image to read from.
            logical_x: X coordinate in logical pixels.
            logical_y: Y coordinate in logical pixels.

        Returns:
            RGB tuple (r, g, b).
        """
        physical_x = int(logical_x * self.scale)
        physical_y = int(logical_y * self.scale)

        # Clamp to image bounds
        physical_x = min(physical_x, screenshot.size[0] - 1)
        physical_y = min(physical_y, screenshot.size[1] - 1)

        pixel = screenshot.getpixel((physical_x, physical_y))
        # Handle RGBA images
        if len(pixel) == 4:
            return pixel[:3]
        return pixel

    def click(self, logical_x: int, logical_y: int, button: str = 'left'):
        """Click at logical coordinates.

        Args:
            logical_x: X coordinate in logical pixels.
            logical_y: Y coordinate in logical pixels.
            button: 'left' or 'right'.
        """
        pyautogui.click(logical_x, logical_y, button=button)

    def logical_to_physical(self, logical_x: int, logical_y: int) -> tuple:
        """Convert logical coordinates to physical coordinates.

        Args:
            logical_x: X coordinate in logical pixels.
            logical_y: Y coordinate in logical pixels.

        Returns:
            Tuple of (physical_x, physical_y).
        """
        return (int(logical_x * self.scale), int(logical_y * self.scale))

    def physical_to_logical(self, physical_x: int, physical_y: int) -> tuple:
        """Convert physical coordinates to logical coordinates.

        Args:
            physical_x: X coordinate in physical pixels.
            physical_y: Y coordinate in physical pixels.

        Returns:
            Tuple of (logical_x, logical_y).
        """
        return (int(physical_x / self.scale), int(physical_y / self.scale))

from PySide6.QtCore import Qt
from montaris.tools.base import BaseTool


class HandTool(BaseTool):
    """Pan/navigation tool. Left-click drag pans the canvas."""
    name = "Hand"
    is_hand = True

    def cursor(self):
        return Qt.OpenHandCursor

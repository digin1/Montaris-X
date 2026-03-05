from PySide6.QtCore import Qt


class BaseTool:
    name = "Base"

    def __init__(self, app):
        self.app = app

    def on_press(self, pos, layer, canvas):
        pass

    def on_move(self, pos, layer, canvas):
        pass

    def on_release(self, pos, layer, canvas):
        pass

    def on_key_press(self, key, canvas):
        pass

    def cursor(self):
        return Qt.CrossCursor

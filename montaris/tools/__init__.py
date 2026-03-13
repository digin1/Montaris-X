import importlib

TOOL_REGISTRY = {
    'Hand': ('montaris.tools.hand', 'HandTool', 'H', 'Navigate'),
    'Select ROI': ('montaris.tools.select', 'SelectTool', 'Q', 'Navigate'),
    'Brush': ('montaris.tools.brush', 'BrushTool', 'B', 'Drawing'),
    'Eraser': ('montaris.tools.eraser', 'EraserTool', 'E', 'Drawing'),
    'Polygon': ('montaris.tools.polygon', 'PolygonTool', 'P', 'Drawing'),
    'Bucket Fill': ('montaris.tools.bucket_fill', 'BucketFillTool', 'G', 'Drawing'),
    'Rectangle': ('montaris.tools.rectangle', 'RectangleTool', 'R', 'Shape'),
    'Circle': ('montaris.tools.circle', 'CircleTool', 'C', 'Shape'),
    'Stamp': ('montaris.tools.stamp', 'StampTool', 'S', 'Shape'),
    'Transform (selected)': ('montaris.tools.transform', 'TransformTool', 'T', 'Transform'),
    'Move (selected)': ('montaris.tools.move', 'MoveTool', 'V', 'Transform'),
    'Transform All': ('montaris.tools.transform', 'TransformAllTool', 'Shift+T', 'Transform'),
    'Move All': ('montaris.tools.move', 'MoveAllTool', 'Shift+V', 'Transform'),
}


def get_tool_class(tool_name):
    """Lazily import and return the tool class for the given tool name."""
    module_path, class_name, _shortcut, _category = TOOL_REGISTRY[tool_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

---
sidebar_position: 6
title: Session Management
description: Auto-save, session restore, and crash recovery in Montaris-X.
keywords: [auto-save, session management, crash recovery, save progress]
---

# Session Management

Montaris-X protects your work with automatic session saving and full crash recovery.

## Save Session

Save your entire workspace — image, all ROI layers, zoom level, and tool settings:

- **Menu:** File → Save Progress (`Ctrl+Shift+S`)
- Creates a session file that captures the complete application state
- Reload any session to pick up exactly where you left off

## Auto-Save

Montaris-X automatically saves your session at regular intervals:

- No configuration needed — it just works
- Auto-save files are stored alongside your working files
- If the application closes unexpectedly, your last auto-save is available on next launch

## Crash Recovery

If Montaris-X detects an unsaved session from a previous crash:

- A recovery dialog appears on startup
- Choose to **restore** the previous session or **discard** it
- Recovery loads all ROI layers, the image, and your last view position

## Save ROI Set

For sharing or archiving just the ROI data (without the image):

- **Menu:** File → Save ROI Set (`Ctrl+S`)
- Saves as `.npz` archive — compact, fast, and lossless
- Load on any machine with Montaris-X

## Best Practices

- Use **Save Session** (`Ctrl+Shift+S`) for work-in-progress
- Use **Save ROI Set** (`Ctrl+S`) for sharing or archiving final annotations
- Export to ImageJ format for interoperability with FIJI workflows

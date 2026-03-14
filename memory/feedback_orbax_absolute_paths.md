---
name: orbax requires absolute paths
description: orbax checkpoint paths must be absolute; resolve relative paths in Python before passing to orbax
type: feedback
---

Always use `Path(...).resolve()` before passing any path to orbax (checkpointer.save / checkpointer.restore). Relative paths silently break because orbax requires absolute paths.

**Why:** orbax quirk — it does not handle relative paths correctly.

**How to apply:** Whenever reading or writing orbax checkpoints with a user-supplied path, call `.resolve()` on the Path object before use. Do this at the Python entry point (e.g. in the training script when reading `cfg.restart_from`), not in shell/justfile.

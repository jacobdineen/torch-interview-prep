"""Maintenance, authoring, and secondary-verification commands.

The daily commands (prep, check, projects) and the two graders (runner,
project_runner) stay at the repo root; these run-occasionally scripts live here.
Each adds the repo root to sys.path so it can be run as `python tools/<name>.py`.
"""

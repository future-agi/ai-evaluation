"""Subprocess ENTRY MODULES for live lanes (P3-D1).

These files are only ever executed as subprocesses
(``sys.executable path/to/worker.py``) by ``_runner.spawn_lane_subprocess``
with the lane extra installed; the release process never imports them. They
are the ONLY sanctioned home for framework imports under the
live_lane_boundary gate (workers here keep even those lazy, inside function
bodies, so the files also import clean without any extra).
"""

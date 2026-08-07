"""
WSGI entry point for hosted deployment (Azure App Service, or any gunicorn host).

Azure's Oryx builder looks for a WSGI callable at the repository root; the
application itself lives in UI/app.py, so this module puts UI/ on the import
path and re-exports it.

Start with a SINGLE worker — see the store note in UI/app.py:

    gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 1 wsgi:app
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "UI"))

from app import app  # noqa: E402  (path must be set before this import)

__all__ = ["app"]

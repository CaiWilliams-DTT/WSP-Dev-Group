"""
Local development launcher. NOT the deployment entry point — hosted
deployments import wsgi.py and are served by gunicorn (see wsgi.py).
"""
import os
import subprocess
import sys

# To Add: Personality Graphic for each style guide [start with blank question mark graphic then after 20 iterations change to graphic that represents the style guide]
# No need to save profiles on creation

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UI", "app.py")

subprocess.run([sys.executable, APP], check=True)

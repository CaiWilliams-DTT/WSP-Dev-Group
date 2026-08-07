#!/bin/bash
# Azure App Service startup command.
# Set this file as the Startup Command (Configuration > General settings),
# or paste the gunicorn line below directly.
#
# --workers 1 is REQUIRED, not a tuning choice: session state lives in an
# in-process dict (see the store note in UI/app.py). More than one worker
# scatters a single browser's requests across processes that do not share
# it. The same applies to scaling out — keep the plan at one instance until
# that state moves somewhere shared.
#
# --timeout 600 exceeds Azure's 230s front-end idle timeout, so a slow Groq
# generation is cut off by the front end with a 502 rather than having the
# worker killed mid-request.
exec gunicorn \
    --bind=0.0.0.0:8000 \
    --workers 1 \
    --timeout 600 \
    --access-logfile '-' \
    --error-logfile '-' \
    wsgi:app

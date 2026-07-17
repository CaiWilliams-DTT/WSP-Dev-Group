import os
import random
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
# Required for secure session cookie signing
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
APP_TITLE = "SREA Dev Group"
APP_ICON = ""
MAIN_SUBTITLE = ""

# =====================================================================
# BACKEND HOOKS
# =====================================================================

def get_next_candidates():
    """Generate or fetch the next pair of items (A and B) for comparison."""
    candidate_a = 'Text A Placeholder'
    candidate_b = 'Text B Placeholder'
    return candidate_a, candidate_b

def update_model_state(choice: str, item_a, item_b):
    """Trigger internal state updates, database writes, or model training."""
    pass

def get_tracking_table() -> pd.DataFrame:
    """Return a Pandas DataFrame summarizing current system state."""
    data = {
        "Metric A": np.random.uniform(0, 1, size=5),
        "Metric B": np.random.uniform(10, 50, size=5),
        "Status": ["Active", "Active", "Pending", "Active", "Archived"]
    }
    return pd.DataFrame(data, index=[f"Item {i+1}" for i in range(5)])

def get_metrics() -> dict:
    """Return a dictionary of key-value pairs to display in the dashboard."""
    return {
        "Metric 1": f"{np.random.uniform(0.6, 1.0):.4f}",
        "Metric 2": f"{np.random.randint(100, 500)}",
        "Metric 3": "Optimal"
    }

# =====================================================================
# FLASK ROUTING & SESSION MANAGEMENT
# =====================================================================

@app.before_request
def initialize_session():
    """Ensure session state variables exist, mimicking st.session_state."""
    if "iteration" not in session:
        session["iteration"] = 1
    if "candidate_a" not in session or "candidate_b" not in session:
        cand_a, cand_b = get_next_candidates()
        session["candidate_a"] = cand_a
        session["candidate_b"] = cand_b
    if "history" not in session:
        session["history"] = []

@app.route("/", methods=["GET"])
def index():
    metrics = get_metrics()
    return render_template(
        "index.html",
        app_title=APP_TITLE,
        app_icon=APP_ICON,
        main_subtitle=MAIN_SUBTITLE,
        iteration=session["iteration"],
        candidate_a=session["candidate_a"],
        candidate_b=session["candidate_b"],
        metrics=metrics
    )

@app.route("/action", methods=["POST"])
def handle_action():
    action = request.form.get("action")
    
    if action == "reset":
        session["iteration"] = 1
        session["history"] = []
        cand_a, cand_b = get_next_candidates()
        session["candidate_a"] = cand_a
        session["candidate_b"] = cand_b
        
    elif action == "select_a":
        update_model_state("A", session["candidate_a"], session["candidate_b"])
        history = session["history"]
        history.append({"Iter": session["iteration"], "Choice": "Candidate A"})
        session["history"] = history
        session["iteration"] += 1
        cand_a, cand_b = get_next_candidates()
        session["candidate_a"] = cand_a
        session["candidate_b"] = cand_b
        
    elif action == "select_b":
        update_model_state("B", session["candidate_a"], session["candidate_b"])
        history = session["history"]
        history.append({"Iter": session["iteration"], "Choice": "Candidate B"})
        session["history"] = history
        session["iteration"] += 1
        cand_a, cand_b = get_next_candidates()
        session["candidate_a"] = cand_a
        session["candidate_b"] = cand_b
        
    elif action == "skip":
        cand_a, cand_b = get_next_candidates()
        session["candidate_a"] = cand_a
        session["candidate_b"] = cand_b

    # Mark session as modified to save mutable structures like lists/dicts
    session.modified = True 
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
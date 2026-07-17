import sys
import streamlit as st
import pandas as pd
import numpy as np

# =====================================================================
# CONFIGURATION & CONSTANTS
# Define global settings or fixed dimensions here.
# =====================================================================
APP_TITLE = "SREA Dev Group"
APP_ICON = ""
MAIN_SUBTITLE = ""

# =====================================================================
# PLACEHOLDER BACKEND HOOKS
# Replace these with your project's custom logic, APIs, or ML models.
# =====================================================================

def get_next_candidates():
    """
    Generate or fetch the next pair of items (A and B) for comparison.
    Returns any data structure (dicts, DataFrames, images, text, arrays).
    """
    # EXAMPLE PLACEHOLDER: Replace with your generation/query logic
    candidate_a = {"id": 101, "name": "Option Alpha", "value": np.random.randint(10, 99)}
    candidate_b = {"id": 102, "name": "Option Beta",  "value": np.random.randint(10, 99)}
    return candidate_a, candidate_b

def update_model_state(choice: str, item_a, item_b):
    """
    Trigger internal state updates, database writes, or model training
    based on the user's selection ('A' or 'B').
    """
    # EXAMPLE PLACEHOLDER: Replace with your state update logic
    pass

def get_tracking_table() -> pd.DataFrame:
    """
    Return a Pandas DataFrame summarizing current system state, 
    scores, rankings, or model parameters.
    """
    # EXAMPLE PLACEHOLDER: Replace with your tracking table logic
    data = {
        "Metric A": np.random.uniform(0, 1, size=5),
        "Metric B": np.random.uniform(10, 50, size=5),
        "Status": ["Active", "Active", "Pending", "Active", "Archived"]
    }
    return pd.DataFrame(data, index=[f"Item {i+1}" for i in range(5)])

def get_metrics() -> dict:
    """
    Return a dictionary of key-value pairs to display in the top dashboard area.
    """
    # EXAMPLE PLACEHOLDER: Replace with your KPI / metric calculations
    return {
        "Metric 1": f"{np.random.uniform(0.6, 1.0):.4f}",
        "Metric 2": f"{np.random.randint(100, 500)}",
        "Metric 3": "Optimal"
    }

# =====================================================================
# MAIN STREAMLIT UI APPLICATION
# =====================================================================

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

    ########################## SESSION STATE INITIALIZATION ##########################
    if "iteration" not in st.session_state:
        st.session_state.iteration = 1
    
    if "candidate_a" not in st.session_state or "candidate_b" not in st.session_state:
        st.session_state.candidate_a, st.session_state.candidate_b = get_next_candidates()
    
    if "history" not in st.session_state:
        st.session_state.history = []


    ########################## HEADER & RESET BAR ##########################
    col_title, col_reset = st.columns([8, 2])
    with col_title:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.markdown(MAIN_SUBTITLE)
    
    with col_reset:
        st.write("")  # Vertical alignment spacing
        if st.button("Reset", type="primary", use_container_width=True):
            st.session_state.iteration = 1
            st.session_state.history = []
            st.session_state.candidate_a, st.session_state.candidate_b = get_next_candidates()
            st.rerun()

    st.divider()


    ########################## SECTION 1: METRICS DASHBOARD ##########################
    metrics = get_metrics()
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(label="Current Iteration", value=f"#{st.session_state.iteration}")
    with col_m2:
        st.metric(label="Metric 1", value=metrics.get("Metric 1", "N/A"))
    with col_m3:
        st.metric(label="Metric 2", value=metrics.get("Metric 2", "N/A"))
    with col_m4:
        st.metric(label="Metric 3", value=metrics.get("Metric 3", "N/A"))

    st.divider()

    ########################### SECTION 2: COMPARISON AREA ##########################
    st.subheader("Select Preferred Option")
    st.markdown("Evaluate the two candidates below and choose the preferred option.")

    col_a, col_space, col_b = st.columns([5, 1, 5], border=False)

    with col_a:
        st.container(border=True)
        st.markdown("### Candidate A")
        
        # Display Candidate A (Replace with st.image, st.json, or st.write depending on data type)
        st.json(st.session_state.candidate_a)
        
        if st.button("👈 Select Candidate A", key="btn_a", use_container_width=True):
            update_model_state("A", st.session_state.candidate_a, st.session_state.candidate_b)
            st.session_state.history.append({"Iter": st.session_state.iteration, "Choice": "Candidate A"})
            st.session_state.iteration += 1
            st.session_state.candidate_a, st.session_state.candidate_b = get_next_candidates()
            st.rerun()

    with col_b:
        st.container(border=True)
        st.markdown("### Candidate B")
        
        # Display Candidate B
        st.json(st.session_state.candidate_b)
        
        if st.button("Select Candidate B 👉", key="btn_b", use_container_width=True):
            update_model_state("B", st.session_state.candidate_a, st.session_state.candidate_b)
            st.session_state.history.append({"Iter": st.session_state.iteration, "Choice": "Candidate B"})
            st.session_state.iteration += 1
            st.session_state.candidate_a, st.session_state.candidate_b = get_next_candidates()
            st.rerun()

    # Skip / Neutral action
    col_skip_1, col_skip_2, col_skip_3 = st.columns([1, 2, 1])
    with col_skip_2:
        if st.button("🤷 No Preference / Skip", use_container_width=True):
            st.session_state.candidate_a, st.session_state.candidate_b = get_next_candidates()
            st.rerun()

    st.divider()


# =====================================================================
# EXECUTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
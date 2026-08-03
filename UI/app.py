import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, session, url_for)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from DIMENTIONS.input import StyleFeatureSpace
from ALGO.pref_learn_algo import MythosLinearAlgo, PairsExhausted
from LLM_API.llm_api import get_llm_placeholder, style_dict_to_guide

app = Flask(__name__)
# Required for secure session cookie signing
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
APP_TITLE = "SREA Dev Group"
MAIN_SUBTITLE = "Active style preference learning"
PROFILE_SCHEMA_VERSION = 1
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
MIN_ANSWERED_FOR_EXPORT = 5

os.makedirs(PROFILES_DIR, exist_ok=True)

# Dev-only diagnostics (see dev_metrics.py). With the flag off the module
# is never imported: the route is not registered (it 404s), no nav link
# renders, and no collection or timing runs in the request path.
DEV_METRICS = os.environ.get("DEV_METRICS", "").strip().lower() in ("1", "true", "yes", "on")
if DEV_METRICS:
    from dev_metrics import MetricsCollector, register_dev_metrics

# =====================================================================
# SERVER-SIDE STATE STORE
#
# The Flask session is a signed cookie: it can only hold small, JSON-
# serialisable values. The model object and the numpy feature vectors are
# neither, so they live here, in-process, keyed by a session id. Only that
# id goes in the cookie. Durable persistence is handled by the profile
# save/load routes below, which write JSON files under profiles/.
# =====================================================================

_STORE = {}
_STORE_LOCK = threading.Lock()


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_state():
    """Build a fresh style space, model, and first comparison pair."""
    style_space = StyleFeatureSpace()
    feature_matrix = style_space.generate_feature_matrix(as_numpy=True)
    algo = MythosLinearAlgo(vectors=feature_matrix, past_scores=False)
    state = {
        "style_space": style_space,
        "algo": algo,
        "iteration": 1,
        "history": [],
        "profile_name": None,
        "profile_slug": None,
        "created": _utc_now(),
        "dirty": False,
        "exhausted": False,
        "metrics": MetricsCollector() if DEV_METRICS else None,
        "metrics_raw": None,
    }
    refresh_candidates(state)
    return state


def get_state():
    """Return the state for the current browser session, creating it if needed."""
    sid = session.get("sid")

    with _STORE_LOCK:
        if sid is None or sid not in _STORE:
            sid = uuid.uuid4().hex
            session["sid"] = sid
            _STORE[sid] = new_state()
        return _STORE[sid]


def replace_state(state):
    """Swap in a new state object for the current browser session."""
    sid = session.get("sid") or uuid.uuid4().hex
    session["sid"] = sid
    with _STORE_LOCK:
        _STORE[sid] = state
    return state


# =====================================================================
# STATE HELPERS
# =====================================================================
def refresh_candidates(state):
    """Draw the next unseen comparison pair; flip to the end state when none remain."""
    algo = state["algo"]
    try:
        if state["metrics"] is not None:
            t0 = time.perf_counter()
            a_vect, b_vect = algo.get_comparison()
            state["metrics"].note_acquisition(time.perf_counter() - t0, algo)
        else:
            a_vect, b_vect = algo.get_comparison()
    except PairsExhausted:
        state["exhausted"] = True
        state["a_vect"] = state["b_vect"] = None
        state["profile_a"] = state["profile_b"] = None
        state["candidate_a"] = state["candidate_b"] = None
        return
    state["exhausted"] = False
    state["a_vect"], state["b_vect"] = a_vect, b_vect
    space = state["style_space"]
    state["profile_a"] = space.devectorize_profile(a_vect)
    state["profile_b"] = space.devectorize_profile(b_vect)
    state["candidate_a"] = get_llm_placeholder(style_dict_to_guide(state["profile_a"]))
    state["candidate_b"] = get_llm_placeholder(style_dict_to_guide(state["profile_b"]))


def record_choice(state, choice: str):
    """Apply the user's preference to the model, log it, and advance."""
    algo = state["algo"]
    if state["metrics"] is not None:
        t0 = time.perf_counter()
        algo.update_score(state["a_vect"], state["b_vect"], choice)
        state["metrics"].record_iteration(algo, outcome=choice,
                                          fit_seconds=time.perf_counter() - t0)
    else:
        algo.update_score(state["a_vect"], state["b_vect"], choice)
    state["history"].append({"Iter": state["iteration"], "Choice": f"Candidate {choice}"})
    state["iteration"] += 1
    state["dirty"] = True
    refresh_candidates(state)


def record_skip(state):
    """Log a no-preference outcome (kept out of the likelihood) and advance."""
    algo = state["algo"]
    algo.record_skip(state["a_vect"], state["b_vect"])
    if state["metrics"] is not None:
        state["metrics"].record_iteration(algo, outcome="SKIP", fit_seconds=0.0)
    state["history"].append({"Iter": state["iteration"], "Choice": "Skipped"})
    state["iteration"] += 1
    state["dirty"] = True
    refresh_candidates(state)


def comparison_counts(algo):
    """(answered, skipped) totals from the comparison log."""
    answered = skipped = 0
    for _a, _b, outcome in algo.past_comparisons:
        if str(outcome).strip().upper() == "SKIP":
            skipped += 1
        else:
            answered += 1
    return answered, skipped


def export_status(state):
    """Whether the style guide export is meaningful yet, and why not if not."""
    answered, _ = comparison_counts(state["algo"])
    if answered >= MIN_ANSWERED_FOR_EXPORT:
        return True, None
    return False, (f"Needs at least {MIN_ANSWERED_FOR_EXPORT} answered comparisons "
                   f"before the ranking is meaningful (currently {answered}).")


def top_ranking(state, limit=10):
    """Best vectors under the fitted utilities, decoded for display."""
    algo = state["algo"]
    space = state["style_space"]
    order = np.argsort(-algo.scores, kind="stable")[:limit]
    return [{
        "rank": rank,
        "vector_id": int(idx),
        "profile": space.devectorize_profile(algo.population[idx]),
        "score": float(algo.scores[idx]),
        "std": float(algo.score_stds[idx]),
    } for rank, idx in enumerate(order, start=1)]


# =====================================================================
# PROFILE PERSISTENCE (JSON files under profiles/, no pickling)
# =====================================================================
class ProfileError(ValueError):
    """User-facing profile failure; always surfaced as a flash, never a 500."""


def slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]
    if not slug:
        raise ProfileError("Profile name must contain at least one letter or digit.")
    return slug


def profile_path(slug):
    """Resolve a slug to a file path, refusing anything outside profiles/."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,59}", slug or ""):
        raise ProfileError("Invalid profile identifier.")
    path = os.path.abspath(os.path.join(PROFILES_DIR, slug + ".json"))
    if os.path.dirname(path) != os.path.abspath(PROFILES_DIR):
        raise ProfileError("Invalid profile identifier.")
    return path


def list_profiles():
    profiles = []
    try:
        names = sorted(os.listdir(PROFILES_DIR))
    except OSError:
        return profiles
    for fname in names:
        if not fname.endswith(".json"):
            continue
        slug = fname[:-5]
        entry = {"slug": slug, "name": slug, "updated": None}
        try:
            with open(os.path.join(PROFILES_DIR, fname), encoding="utf-8") as fh:
                data = json.load(fh)
            entry["name"] = data.get("name") or slug
            entry["updated"] = data.get("updated")
        except (OSError, ValueError):
            entry["name"] = f"{slug} (unreadable)"
        profiles.append(entry)
    return profiles


def serialize_state(state):
    data = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "name": state["profile_name"],
        "slug": state["profile_slug"],
        "created": state["created"],
        "updated": _utc_now(),
        "feature_space": state["style_space"].features,
        "iteration": state["iteration"],
        "history": state["history"],
        "algo": state["algo"].to_dict(),
    }
    if state["metrics"] is not None:
        data["dev_metrics"] = state["metrics"].to_dict()
    elif state.get("metrics_raw") is not None:
        # Preserve diagnostics captured under DEV_METRICS even though the
        # flag is off in this process.
        data["dev_metrics"] = state["metrics_raw"]
    return data


def save_profile(state):
    data = serialize_state(state)
    path = profile_path(state["profile_slug"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    state["dirty"] = False
    return data


def restore_state(data):
    """Rebuild session state from a profile file's contents."""
    version = data.get("schema_version")
    if version != PROFILE_SCHEMA_VERSION:
        raise ProfileError(f"Unsupported profile schema version {version!r} "
                           f"(expected {PROFILE_SCHEMA_VERSION}).")
    try:
        style_space = StyleFeatureSpace(features=data["feature_space"])
        algo = MythosLinearAlgo.from_dict(data["algo"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ProfileError(f"Profile data is malformed: {exc}") from exc
    state = {
        "style_space": style_space,
        "algo": algo,
        "iteration": int(data.get("iteration", 1)),
        "history": list(data.get("history", [])),
        "profile_name": data.get("name"),
        "profile_slug": data.get("slug"),
        "created": data.get("created", _utc_now()),
        "dirty": False,
        "exhausted": False,
        "metrics": None,
        "metrics_raw": None,
    }
    if DEV_METRICS:
        # Tolerates an absent or foreign-version dev_metrics key.
        state["metrics"] = MetricsCollector.from_dict(data.get("dev_metrics"))
    else:
        state["metrics_raw"] = data.get("dev_metrics")
    refresh_candidates(state)
    return state


# =====================================================================
# TEMPLATE CHROME (header shared by every page)
# =====================================================================
@app.context_processor
def inject_chrome():
    state = get_state()
    return {
        "app_title": APP_TITLE,
        "main_subtitle": MAIN_SUBTITLE,
        "profiles": list_profiles(),
        "active_profile": state.get("profile_name"),
        "unsaved_changes": bool(state.get("dirty")),
        "dev_metrics_enabled": DEV_METRICS,
    }


# =====================================================================
# FLASK ROUTING
# =====================================================================
@app.route("/", methods=["GET"])
def index():
    state = get_state()
    algo = state["algo"]
    answered, skipped = comparison_counts(algo)
    export_ok, export_reason = export_status(state)
    return render_template(
        "index.html",
        iteration=state["iteration"],
        exhausted=state["exhausted"],
        candidate_a=state["candidate_a"],
        candidate_b=state["candidate_b"],
        profile_a=state["profile_a"],
        profile_b=state["profile_b"],
        answered=answered,
        skipped=skipped,
        pairs_remaining=algo.n_pairs_total - algo.n_pairs_presented,
        ranking=top_ranking(state) if state["exhausted"] else None,
        export_ok=export_ok,
        export_reason=export_reason,
    )


@app.route("/action", methods=["POST"])
def handle_action():
    action = request.form.get("action")
    state = get_state()

    if action == "reset":
        # Discard the model entirely so the posterior starts clean
        replace_state(new_state())
        flash("Session reset — the posterior starts clean.", "info")
    elif state["exhausted"]:
        flash("No comparisons left — every pair has been presented.", "error")
    elif action == "select_a":
        record_choice(state, "A")
    elif action == "select_b":
        record_choice(state, "B")
    elif action == "skip":
        record_skip(state)

    return redirect(url_for("index"))


@app.route("/profiles", methods=["GET"])
def profiles_index():
    return jsonify(list_profiles())


@app.route("/profile/new", methods=["POST"])
def profile_new():
    name = (request.form.get("name") or "").strip()
    try:
        if not name:
            raise ProfileError("Enter a name for the new profile.")
        slug = slugify(name)
        path = profile_path(slug)
        if os.path.exists(path):
            raise ProfileError(f'A profile called "{name}" already exists — '
                               f'load it or pick another name.')
        state = new_state()
        state["profile_name"] = name
        state["profile_slug"] = slug
        replace_state(state)
        save_profile(state)
    except ProfileError as exc:
        flash(str(exc), "error")
    except OSError as exc:
        flash(f"Could not write the profile file: {exc}", "error")
    else:
        flash(f'Created profile "{name}".', "success")
    return redirect(url_for("index"))


@app.route("/profile/save", methods=["POST"])
def profile_save():
    state = get_state()
    if not state.get("profile_slug"):
        flash("No active profile — create one first.", "error")
        return redirect(url_for("index"))
    try:
        save_profile(state)
    except (ProfileError, OSError) as exc:
        flash(f"Save failed: {exc}", "error")
    else:
        flash(f'Saved profile "{state["profile_name"]}".', "success")
    return redirect(url_for("index"))


@app.route("/profile/load", methods=["POST"])
def profile_load():
    slug = (request.form.get("slug") or "").strip()
    if not slug:
        flash("Choose a profile to load.", "error")
        return redirect(url_for("index"))
    try:
        path = profile_path(slug)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        state = restore_state(data)
    except FileNotFoundError:
        flash("That profile file no longer exists.", "error")
    except json.JSONDecodeError:
        flash("That profile file is not valid JSON — it may be corrupted.", "error")
    except ProfileError as exc:
        flash(str(exc), "error")
    except Exception as exc:  # never surface a 500 for a bad profile file
        flash(f"Could not load the profile: {exc}", "error")
    else:
        replace_state(state)
        flash(f'Loaded profile "{state["profile_name"] or slug}".', "success")
    return redirect(url_for("index"))


@app.route("/export", methods=["GET"])
def export_styleguide():
    state = get_state()
    export_ok, export_reason = export_status(state)
    if not export_ok:
        flash(export_reason, "error")
        return redirect(url_for("index"))

    algo = state["algo"]
    space = state["style_space"]
    # Highest fitted utility, not raw win counts.
    best = int(np.argmax(algo.scores))
    best_profile = space.devectorize_profile(algo.population[best])
    answered, skipped = comparison_counts(algo)
    profile_label = state["profile_name"] or "Unsaved session"
    now = datetime.now(timezone.utc)

    markdown = "\n".join([
        f"# Style guide — {profile_label}",
        "",
        f"- Generated: {now.replace(microsecond=0).isoformat()}",
        f"- Answered comparisons: {answered} (plus {skipped} skipped)",
        f"- Top vector: #{best} "
        f"(utility {algo.scores[best]:+.3f} ± {algo.score_stds[best]:.3f})",
        "",
        style_dict_to_guide(best_profile),
        "",
    ])
    filename = (f"{state['profile_slug'] or 'session'}_styleguide_"
                f"{now.strftime('%Y%m%d-%H%M%S')}.md")
    return Response(
        markdown,
        mimetype="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if DEV_METRICS:
    register_dev_metrics(app, get_state)


if __name__ == "__main__":
    # use_reloader=False keeps the in-process store from being wiped on file saves
    app.run(debug=True, port=5000, use_reloader=False)

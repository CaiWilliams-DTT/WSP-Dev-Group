import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from flask import (Flask, Response, flash, has_request_context, redirect,
                   render_template, request, session, url_for)
from werkzeug.middleware.proxy_fix import ProxyFix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from DIMENTIONS.input import StyleFeatureSpace
from ALGO.pref_learn_algo import MythosLinearAlgo, PairsExhausted
from LLM_API.llm_api import get_llm, get_llm_placeholder, pick_prompt, style_dict_to_guide

# Azure App Service sets WEBSITE_SITE_NAME in every hosted container, so a
# deployment is recognised as production without anyone remembering to set a
# flag. APP_ENV overrides it in both directions for other hosts.
IS_PRODUCTION = os.environ.get(
    "APP_ENV", "production" if os.environ.get("WEBSITE_SITE_NAME") else "development"
).strip().lower() == "production"

app = Flask(__name__)

# Required for secure session cookie signing. A shared, known key lets anyone
# forge a session cookie and adopt another user's server-side state, so in
# production a real key is mandatory rather than defaulted.
_secret = os.environ.get("SECRET_KEY")
if not _secret:
    if IS_PRODUCTION:
        raise RuntimeError(
            "SECRET_KEY is not set. Set it as an application setting before "
            "serving traffic — without it session cookies are forgeable."
        )
    _secret = "dev-secret-key-change-in-prod"
app.secret_key = _secret

# TLS terminates at the Azure front end and the request reaches us over plain
# HTTP, so without this Flask sees the wrong scheme and client address.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # Only over HTTPS in production; setting it in local dev would stop the
    # cookie being stored at all over http://localhost.
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
)

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
APP_TITLE = "SREA Dev Group"
MAIN_SUBTITLE = "Active style preference learning"
PROFILE_SCHEMA_VERSION = 1
# Overridable because the deployed application directory is read-only under
# WEBSITE_RUN_FROM_PACKAGE and is replaced wholesale on every redeploy. On
# Azure point this at the persistent share, e.g. /home/data/profiles.
PROFILES_DIR = os.environ.get("PROFILES_DIR") or os.path.join(BASE_DIR, "profiles")
MIN_ANSWERED_FOR_EXPORT = 5
# Profiles are uploaded from the user's own machine, so cap what we will
# read. The Flask-level limit rejects the request outright (413, handled
# below); the smaller one produces a friendly message for a plausible file.
MAX_PROFILE_BYTES = 2 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

# A read-only application directory must not take the whole app down at
# import: everything except profile save/load still works, and those routes
# already report OSError to the user as a flash.
try:
    os.makedirs(PROFILES_DIR, exist_ok=True)
except OSError as _exc:
    print(f"warning: profile directory {PROFILES_DIR} is not writable ({_exc}); "
          f"saving profiles will fail. Set PROFILES_DIR to a writable path.",
          file=sys.stderr)

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
#
# IMPORTANT — this store is per-process and per-instance. The app must run
# with a SINGLE gunicorn worker on a SINGLE instance: with more, requests
# from one browser land on processes that do not share _STORE and the user
# loses their model mid-session. Do not add --workers, and do not scale out
# or enable autoscale, without first moving this state to something shared
# (Redis, or a database). Entries are dropped after SESSION_TTL_SECONDS of
# inactivity, and the oldest are shed past MAX_SESSIONS, so a stream of
# visitors cannot grow the process without bound.
# =====================================================================

_STORE = {}
_STORE_LOCK = threading.Lock()
_LAST_SEEN = {}

SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 4 * 60 * 60))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", 500))


def _touch(sid):
    """Mark a session as active. Caller must hold _STORE_LOCK."""
    _LAST_SEEN[sid] = time.monotonic()


def _evict(protect=None):
    """Drop idle and surplus sessions. Caller must hold _STORE_LOCK.

    Unsaved work is lost when a session is dropped, which is the same
    outcome as the app instance recycling — the profile files are the
    durable copy. `protect` is the session being served right now, which
    must survive the sweep regardless.
    """
    now = time.monotonic()
    for sid in [s for s, seen in _LAST_SEEN.items()
                if s != protect and now - seen > SESSION_TTL_SECONDS]:
        _STORE.pop(sid, None)
        _LAST_SEEN.pop(sid, None)
    if len(_STORE) > MAX_SESSIONS:
        oldest = sorted(_LAST_SEEN.items(), key=lambda kv: kv[1])
        for sid, _seen in oldest[:len(_STORE) - MAX_SESSIONS]:
            if sid == protect:
                continue
            _STORE.pop(sid, None)
            _LAST_SEEN.pop(sid, None)


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def new_state(api_key=None):
    """Build a fresh style space, model, and first comparison pair.

    `api_key` is carried in rather than reset because the key belongs to the
    person at the browser, not to the model being trained — a session reset
    or a profile switch should not make them type it again.
    """
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
        # Server-side only: never goes in the session cookie (which is signed,
        # not encrypted) and never gets written into a profile file.
        "api_key": api_key,
        "llm_error": None,
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
        _touch(sid)
        _evict(protect=sid)
        return _STORE[sid]

def replace_state(state):
    """Swap in a new state object for the current browser session."""
    sid = session.get("sid") or uuid.uuid4().hex
    session["sid"] = sid
    with _STORE_LOCK:
        _STORE[sid] = state
        _touch(sid)
    return state


# =====================================================================
# STATE HELPERS
# =====================================================================
def generate_sample(state, style_guide, prompt):
    """
    Produce the sample text for one style profile.

    `prompt` is the writing task, drawn once per pair by the caller so both
    candidates answer the same task and the user is comparing style alone.

    With no user-supplied key the app stays on the offline placeholder, so
    ordinary use costs nothing and never reaches Groq. Entering a key on the
    diagnostics page opts that session in to live generation, billed to that
    key instead of the deployment's. A key that is rejected or rate-limited
    must not break the comparison loop, so failures fall back to the
    placeholder and are reported once per pair.
    """
    api_key = state.get("api_key")
    if not api_key:
        return get_llm_placeholder(style_guide, prompt=prompt)
    try:
        text = get_llm(style_guide, api_key=api_key, prompt=prompt)
    except Exception as exc:
        # Scrub the key in case the client echoed it back in the message.
        detail = str(exc).replace(api_key, "[key]")
        state["llm_error"] = f"{type(exc).__name__}: {detail}"
        return get_llm_placeholder(style_guide, prompt=prompt)
    return text


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
    state["llm_error"] = None
    # One task for the whole pair, so the two candidates differ only in style.
    prompt = pick_prompt()
    state["candidate_a"] = generate_sample(state, style_dict_to_guide(state["profile_a"]), prompt)
    state["candidate_b"] = generate_sample(state, style_dict_to_guide(state["profile_b"]), prompt)
    if state["llm_error"] and has_request_context():
        flash(f"Generation with your API key failed, showing placeholder text "
              f"— {state['llm_error']}", "error")

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

def restore_state(data, api_key=None):
    """Rebuild session state from a profile file's contents.

    `api_key` is supplied by the caller, never read from the file: profiles
    are uploaded by users and must not be able to inject a credential.
    """
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
        "api_key": api_key,
        "llm_error": None,
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
        answered=answered,
        skipped=skipped,
        confidence=100.0 * algo.optimality_confidence(),
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
        replace_state(new_state(api_key=state.get("api_key")))
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


@app.route("/profile/new", methods=["POST"])
def profile_new():
    name = (request.form.get("name") or "").strip()
    api_key = get_state().get("api_key")
    try:
        if not name:
            raise ProfileError("Enter a name for the new profile.")
        slug = slugify(name)
        path = profile_path(slug)
        if os.path.exists(path):
            raise ProfileError(f'A profile called "{name}" already exists — '
                               f'load it or pick another name.')
        state = new_state(api_key=api_key)
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
    """
    Load a profile from a file the user picks on their own machine.

    Deliberately an upload rather than a server-side pick list: this runs as
    a shared web app, so one person's results must never be discoverable —
    let alone loadable — by another. The server therefore never enumerates
    profiles/ for the browser; you can only open a file you already hold.
    """
    upload = request.files.get("profile")
    if upload is None or not upload.filename:
        flash("Choose a profile file to load.", "error")
        return redirect(url_for("index"))
    api_key = get_state().get("api_key")
    try:
        raw = upload.read(MAX_PROFILE_BYTES + 1)
        if len(raw) > MAX_PROFILE_BYTES:
            raise ProfileError("That file is too large to be a profile "
                               f"(limit {MAX_PROFILE_BYTES // (1024 * 1024)} MB).")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ProfileError("That file does not contain a profile object.")
        state = restore_state(data, api_key=api_key)
    except UnicodeDecodeError:
        flash("That file is not UTF-8 text — profiles are JSON.", "error")
    except json.JSONDecodeError:
        flash("That file is not valid JSON — it may be corrupted.", "error")
    except ProfileError as exc:
        flash(str(exc), "error")
    except Exception as exc:  # never surface a 500 for a bad profile file
        flash(f"Could not load the profile: {exc}", "error")
    else:
        replace_state(state)
        label = state["profile_name"] or os.path.basename(upload.filename)
        flash(f'Loaded profile "{label}".', "success")
    return redirect(url_for("index"))


@app.errorhandler(413)
def profile_upload_too_large(_exc):
    """Oversized upload: a flash beats Flask's default error page."""
    flash("That file is too large to be a profile.", "error")
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
    # Local development only — in production this file is imported by wsgi.py
    # and served by gunicorn, which never runs this block.
    #
    # debug is force-disabled under production because the Werkzeug debugger
    # exposes an interactive console: reachable from the internet, it is
    # remote code execution.
    debug = not IS_PRODUCTION
    # use_reloader=False keeps the in-process store from being wiped on file saves
    app.run(host="127.0.0.1", port=5000, debug=debug, use_reloader=False)

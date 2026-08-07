# Deploying to Azure App Service

Linux App Service, Python 3.13 runtime.

## 1. Application settings

Configuration > Environment variables. These are the whole configuration —
`.env` is deliberately ignored when running on App Service, so anything not
set here is not set at all.

| Setting | Value | Why |
| --- | --- | --- |
| `SECRET_KEY` | a fresh random string | Signs session cookies. **The app refuses to start without it.** Generate with `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `PROFILES_DIR` | `/home/data/profiles` | `/home` is the persistent share. The default location inside the app directory is read-only under `WEBSITE_RUN_FROM_PACKAGE` and is wiped on every redeploy. |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `1` | Makes the platform install `requirements.txt`. |
| `GROQ_API_KEY` | *(optional)* | Only needed if the deployment should have a fallback key. Users supplying their own key on the diagnostics page are billed to their own account instead. |
| `DEV_METRICS` | `true` | Registers `/dev/metrics`. This is the on/off switch — unset it and those routes 404 with no collection overhead anywhere. See the diagnostics section below. |
| `DEV_METRICS_LOO_EVERY` | *(optional, default 10)* | Compute leave-one-out accuracy every Nth comparison. `1` restores a value on every iteration, at the cost described below. |
| `DEV_METRICS_LOO_MAX` | `80` | Skip LOO entirely past this many answered comparisons. Bounds the worst-case stall; the default of 200 allows a 5-second one. |
| `SESSION_TTL_SECONDS` | *(optional, default 14400)* | Idle sessions dropped after this long. |
| `MAX_SESSIONS` | *(optional, default 500)* | Oldest sessions shed past this count. |

`APP_ENV` is not normally needed: App Service sets `WEBSITE_SITE_NAME`, which
the app treats as the production signal. Set `APP_ENV=development` to opt out,
or `APP_ENV=production` to opt in on a non-Azure host.

## 2. Startup command

Configuration > General settings > Startup Command:

```
startup.sh
```

Or paste the equivalent directly:

```
gunicorn --bind=0.0.0.0:8000 --workers 1 --timeout 600 wsgi:app
```

**`--workers 1` is a correctness requirement, not tuning.** Session state
(the fitted model and its numpy vectors) lives in a per-process dict, because
it is not JSON-serialisable and cannot go in the cookie. A second worker, or a
second instance, scatters one browser's requests across processes that do not
share that dict, and the user loses their model mid-session. For the same
reason: **do not enable autoscale or scale out past one instance** without
first moving that state to Redis or a database.

## 3. What must not be deployed

`.env` is gitignored, so a git-based deploy excludes it. If you deploy by
zipping the folder, exclude it explicitly along with `.venv/` and
`__pycache__/` — it contains a live Groq key.

The committed sample profiles under `profiles/` (`test.json`, `test1.json`,
and so on) are harmless — nothing enumerates that directory for the browser —
but they are dead weight and worth deleting.

## 4. Diagnostics in production

`/dev/metrics` and `/dev/metrics.json` are enabled in this deployment and are
**not access-controlled** — anyone with the URL can reach them. Two things
follow from that, both accepted deliberately:

- The pages show model internals (posterior, feature space, LOO accuracy).
  They are per-session: `_state_getter()` returns only the caller's own state,
  so no one can read another person's model through them.
- `/dev/api-key` is a public form inviting a Groq API key. The key is held in
  server-side session state only, never in the cookie, never written to a
  profile, and only ever rendered masked — but it is a credential form on a
  public URL, which is worth knowing about if the app URL circulates widely.

Turn the whole thing off by removing the `DEV_METRICS` setting: the module is
never imported, the routes 404, the nav link disappears, and no collection or
timing runs in the request path.

**Cost.** Collection adds nothing measurable to a normal comparison click
(~60ms with the flag on or off — that baseline is the BALD acquisition in
`ALGO/pref_learn_algo.py`, not diagnostics). The exception is leave-one-out
accuracy, which refits the learner once per answered comparison and so grows
quadratically. It runs on one click in `DEV_METRICS_LOO_EVERY`, and that click
stalls:

| Answered comparisons | Stall on the sampled click |
| --- | --- |
| 40 | ~330 ms |
| 80 | ~1.3 s |
| 120 | ~2.6 s |
| 160 | ~5.1 s |
| past `DEV_METRICS_LOO_MAX` | skipped entirely |

Because the app runs one worker, that stall blocks *every* user, not just the
one who triggered it — which is why `DEV_METRICS_LOO_MAX` is set to 80 above.
The chart plots only the sampled points and skips the gaps; the summary tile
shows the most recent measurement and the iteration it came from.

## 5. Known limits

- **Sessions do not survive a restart.** App Service recycles containers on
  deploy, on scale operations, and after idle timeout. Anyone mid-comparison
  loses their unsaved model; saved profile files under `/home/data` persist.
  Tell users to save, or move the store off-process.
- **Two Groq generations run synchronously per comparison.** Azure's front end
  drops connections idle for 230s, so a stalled provider surfaces as a 502.
- **Profiles are user-held files.** The app never lists `profiles/` to the
  browser; loading is an upload. That is intentional — it stops one person's
  results being reachable by another — so the server-side directory is really
  only a scratch/save area.

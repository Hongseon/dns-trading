# Render Keepalive Setup

Use the GitHub Actions workflow `.github/workflows/keep_render_awake.yml` to
keep the free Render web service awake between Kakao requests.

## Required Secret

Add this repository secret in GitHub:

- `RENDER_HEALTHCHECK_URL`: full health-check URL, for example
  `https://your-service.onrender.com/health`
- `RENDER_WARMUP_URL` (optional): explicit warmup URL, for example
  `https://your-service.onrender.com/warmup`

## What It Does

- Calls `GET /health` every 5 minutes.
- Calls `GET /warmup` right after the health ping so the RAG chain is ready,
  not just the web process.
- Prevents the 15-minute Render free-tier idle sleep from triggering.
- Reduces first-request failures from KakaoTalk skill timeouts.
- Retries timeouts and transient network errors long enough to survive Render
  cold starts. GitHub Actions `exit code 28` typically means the `curl`
  request hit its timeout window before Render finished waking up.

If `RENDER_WARMUP_URL` is not configured, the workflow automatically derives
it by replacing the trailing `/health` in `RENDER_HEALTHCHECK_URL` with
`/warmup`.

## Validation

1. Trigger the workflow manually once from GitHub Actions.
2. Confirm both `/health` and `/warmup` return HTTP 200.
3. Let the service sit idle for 20+ minutes.
4. Send the first Kakao question and confirm there is no skill timeout.
5. Check Render logs for periodic `/health`, `/warmup`, and `skill/query ... callback=...`.

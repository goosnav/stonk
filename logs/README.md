# Runtime logs

- `audit-live.jsonl` / `audit-paper.jsonl`: rotating structured event logs.
- `runtime-live.log`: server stdout/stderr when launched by the app or restart script.

The SQLite `audit` table remains the source of truth. These files are the
grep-friendly debugging mirror and rotate before growing without bound.

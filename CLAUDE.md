# CLAUDE.md

## Project
Marine map gap-filling POC.

## Goal
Build a Python prototype that fills missing values in marine maps with high accuracy.

## Scope
- Python only.
- No server.
- No API design.
- No production security hardening.
- Focus on fast iteration and reproducible experiments.

## Success metrics
- RMSE.
- MAE.
- Coverage of missing regions.
- Visual similarity to ground truth.

## Non-goals
- Production deployment.
- Authentication.
- Scaling infrastructure.
- Real-time guarantees.

## Working style
- Prefer concise, actionable responses.
- Make explicit assumptions.
- Use small, testable steps.
- Keep code minimal and reproducible.
- Save deliverables to `output/`.

## Domain notes
- Common data formats: NetCDF, xarray, NumPy arrays.
- Typical tasks: load data, downsample, create masks, train a simple baseline, evaluate on held-out maps.
- Keep dataset provenance and preprocessing documented.

## Regulatory notes
- Treat the system as decision-support, not an autonomous safety system.
- Document data sources, model versions, evaluation results, and intended limitations.
- Minimize sensitive data and unnecessary retention.

## Output policy
- Save final artifacts to `output/`.
- Prefer CSV, PNG, MD, YAML, or JSON when relevant.

## שפה ותצוגה
- כתוב תמיד בעברית תקנית בלבד.
- שמור על ניסוח ברור, קצר, ומסודר מימין לשמאל.
- השתמש בשורות קצרות וברווחים ברורים כדי לשפר את התצוגה.
- הימנע ככל האפשר משילוב של אנגלית ומספרים בתוך משפט אחד.
- אם יש צורך במונחים באנגלית, הפרד אותם לשורה נפרדת או כתוב אותם בזהירות כדי למנוע שיבוש בתצוגה.

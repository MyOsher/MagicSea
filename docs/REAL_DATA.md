# הרצה על דאטה אמיתי מ-Copernicus

> ⚠️ **חשוב:** את השלבים האלה מריצים **במחשב שלך**, לא בסביבת ה-web של הסוכן —
> מדיניות הרשת שם חוסמת גישה לשרתי הדאטה הימי. הקוד כבר מוכן; זו רק הרצה מקומית.

## שלב 1 — הרשמה (חינם, פעם אחת)
1. נרשמים ב-https://data.marine.copernicus.eu/register
2. מאשרים את המייל ובוחרים שם משתמש + סיסמה.

## שלב 2 — הגדרת פרטי גישה
```bash
cp .env.example .env
# פותחים את .env וממלאים:
#   COPERNICUSMARINE_SERVICE_USERNAME=...
#   COPERNICUSMARINE_SERVICE_PASSWORD=...
```
הקובץ `.env` כבר ב-`.gitignore` — הסיסמה לא תיכנס ל-git.

## שלב 3 — התקנה
```bash
pip install -r requirements.txt
```

## שלב 4 — הורדה
```bash
python -m src.data.download        # מוריד VHM0 לפי config.yaml אל data/raw/
```

## שלב 5 — הרצה מלאה בפקודה אחת
כל הצינור (טעינה → אימון → הערכה → דוח) בפקודה אחת:
```bash
python -m src.run_real data/raw/<file>.nc --epochs 40
```
התוצאות האמיתיות נשמרות בנפרד תחת **`output/real/`** (כדי לא לדרוס את
הסמוק-טסט הסינתטי): `RESULTS_REAL.md`, `evaluation.json`,
`evaluation_example.png`, `train_loss.png`. הדוח מוטבע עם מקור הקובץ,
האזור, טווח התאריכים ואחוז החורים — לשקיפות מלאה.

> אפשר גם ידנית שלב-אחר-שלב: `src.data.load` → `src.inspect_data` →
> `src.train` → `src.evaluate` (ראו README). `run_real` פשוט משרשר אותם.

## מה כבר מטופל בקוד עבור קבצים אמיתיים
`src/data/load.py` עמיד למוזרויות של קבצי Copernicus אמיתיים:
- מימד `depth` יחיד — מוסר אוטומטית.
- ערכי `_FillValue` / masked arrays — הופכים ל-`NaN` (יבשה/פערים).
- שם משתנה ב-case שונה (`VHM0` מול `vhm0`).
- גריד קטן או גדול מ-64×64 — משתנה גודל בשני הכיוונים (ממוצע-בלוקים בהקטנה, nearest בהגדלה).

## כמה דאטה להוריד
לאימון אמיתי כדאי **חודשים** של נתונים, לא שבועיים, כדי לקבל מפות מגוונות
(סערות, מצבי ים שונים). התחילי מ-3–6 חודשים; אפשר להרחיב את טווח התאריכים
ב-`config.yaml` תחת `time.start` / `time.end`.

## התאמות אפשריות ב-`config.yaml`
- `region` — לשנות את תיבת האזור (מומלץ להתחיל קטן = הורדה מהירה).
- `time.start` / `time.end` — טווח התאריכים.
- `variable.copernicus_dataset_id` — לוודא מול הקטלוג שה-id מדויק ופעיל.
- `preprocess.target_size` — רזולוציית היעד (64 = מהיר; אפשר להעלות בהמשך).

## אם אין לך חשבון Copernicus ורוצים דאטה אמיתי בכל זאת
יש שרתי ERDDAP ציבוריים (למשל של NOAA) שמספקים גובה גלים (WaveWatch III)
ללא התחברות, בפורמט NetCDF. אפשר להוריד קובץ ידנית מדפדפן ואז להריץ עליו את
`src.data.load` כרגיל — הצינור זהה.

# MagicSea — Marine Map Gap-Filling POC

מטרת ה-POC: קוד פייתון שמקבל מפת ים עם חורים (ענן/נתון חסר) ומחזיר מפה שלמה בדיוק גבוה.

**פוקוס:** משתנה אחד — **גובה גלים (VHM0, מטרים)**. משימה אחת — השלמת חורים מלאכותיים.
מדד ראשי — **ירידה ב-RMSE מול קו בסיס של אינטרפולציה**.

התוכנית לפי שבועות:

| שבוע | מטרה | קבצים | סטטוס |
|------|------|-------|-------|
| **1** | דאטה ימי נקי כמערך NumPy | `src/data/` , `src/inspect_data.py` | ✅ |
| **2** | חורים מלאכותיים + קו בסיס + RMSE/MAE | `src/masks.py`, `src/baseline.py`, `src/metrics.py`, `src/run_baseline.py` | ✅ |
| **3-4** | מודל U-Net + אימון ראשוני | `src/model.py`, `src/dataset.py`, `src/train.py` | ✅ |
| **5** | הערכה: U-Net מול הבסיס על מפות held-out | `src/evaluate.py` | ✅ |

**תוצאה עיקרית (דאטה סינתטי):** ה-U-Net הוריד RMSE מ-**0.178** ל-**0.126** מ׳ — שיפור של **~29%** מול אינטרפולציה, על 50 מפות שלא נראו באימון. פירוט מלא: [`output/RESULTS.md`](output/RESULTS.md).

---

## התקנה
```bash
pip install -r requirements.txt
```

## הרצה מלאה (מסלול A — בלי חשבון Copernicus)
דאטה סינתטי שמדמה מפות גובה גלים (כולל NaN על יבשה), באותו מבנה כמו הדאטה האמיתי:
```bash
python -m src.data.synthetic                                    # data/raw/mediterranean_wave_SYNTHETIC.nc
python -m src.data.load data/raw/mediterranean_wave_SYNTHETIC.nc # NetCDF -> NumPy 64x64
python -m src.inspect_data data/processed/mediterranean_wave_SYNTHETIC_64x64.npy
python -m src.run_baseline                                       # קו בסיס + RMSE/MAE + תמונת השוואה
python -m src.train --epochs 40                                  # מאמן את ה-U-Net
python -m src.evaluate                                           # U-Net מול הבסיס על מפות held-out
```
> לאימון עדיף סט גדול יותר: `python -m src.data.synthetic --days 200` לפני `load`.

## מסלול B — דאטה אמיתי מ-Copernicus
מדריך מלא צעד-אחר-צעד: **[`docs/REAL_DATA.md`](docs/REAL_DATA.md)**. בקצרה:
```bash
cp .env.example .env            # למלא שם משתמש/סיסמה של Copernicus
python -m src.data.download     # מוריד VHM0 לפי config.yaml
python -m src.data.load data/raw/<file>.nc
python -m src.train --epochs 40
python -m src.evaluate
```
> `load.py` כבר עמיד למוזרויות של קבצים אמיתיים (מימד depth, ערכי fill, גריד בכל גודל).
> את ההורדה עצמה יש להריץ **מקומית** — סביבת ה-web חוסמת גישה לשרתי הדאטה.

## מה קורה מאחורי הקלעים

**שבוע 1 — דאטה:**
- `config.yaml` — מקום אחד לכל ההגדרות: אזור, טווח תאריכים, משתנה (VHM0), רזולוציית יעד (64×64).
- `download.py` — מוריד אזור קטן לשבועיים בעזרת ה-toolbox הרשמי.
- `synthetic.py` — חלופה שמייצרת NetCDF זהה במבנה (גלים במטרים + יבשה כ-NaN), כדי לא להיחסם.
- `load.py` — פותח NetCDF, **מוריד רזולוציה ל-64×64**, שומר `.npy`. חורים/יבשה נשמרים כ-`NaN` בכוונה.
- `inspect_data.py` — סטטיסטיקות + תצוגה מקדימה ל-`output/`.

**שבוע 2 — חורים, בסיס ומדדים:**
- `masks.py` — מייצר חורים מלאכותיים (מלבנים אקראיים או "עננים" בלתי-סדירים). מסתיר רק פיקסלים שהיו תקינים.
- `baseline.py` — ממלא חורים באינטרפולציה (linear + nearest). זה המספר שה-U-Net חייב לנצח.
- `metrics.py` — RMSE ו-MAE, מחושבים **רק על אזור החורים** שהוסתר.
- `run_baseline.py` — מריץ על כל המפות ושומר `output/baseline_metrics.json` + `output/baseline_example.png`.

**שבוע 3–5 — מודל, אימון והערכה:**
- `model.py` — U-Net קומפקטי (2 ערוצי קלט: מפה-עם-חורים + מסכת פיקסלים תקינים).
- `dataset.py` — מייצר זוגות אימון תוך כדי ריצה; מנרמל; חורים אקראיים לאימון וקבועים לוולידציה.
- `train.py` — לולאת אימון (MSE על פיקסלי ים), שומר `output/unet.pt`, `norm_stats.json`, `train_loss.png`.
- `evaluate.py` — משווה U-Net מול הבסיס על **אותן** מפות held-out ו**אותם** חורים; שומר `evaluation.json` + `evaluation_example.png`.

## פלטים
- `data/processed/*_64x64.npy` — מערך `(ימים, 64, 64)` מוכן למודל.
- `output/RESULTS.md` — דוח תוצאות קצר (הדליברבל המרכזי).
- `output/evaluation.json` + `evaluation_example.png` — U-Net מול הבסיס.
- `output/train_loss.png` — עקומת אימון.
- `output/baseline_metrics.json` — RMSE/MAE של הבסיס בלבד.

> **תוצאות דאטה סינתטי:** baseline RMSE ≈ 0.18 מ׳, U-Net RMSE ≈ 0.13 מ׳ (שיפור ~29%).
> אלה מספרים על דאטה מדומה (smoke test) — יש להריץ מחדש על נתוני Copernicus אמיתיים לפני ציטוט ללקוחות.

# MagicSea — Marine Map Gap-Filling POC

מטרת ה-POC: קוד פייתון שמקבל מפת ים עם חורים (ענן/נתון חסר) ומחזיר מפה שלמה בדיוק גבוה.

**פוקוס:** משתנה אחד — **גובה גלים (VHM0, מטרים)**. משימה אחת — השלמת חורים מלאכותיים.
מדד ראשי — **ירידה ב-RMSE מול קו בסיס של אינטרפולציה**.

התוכנית לפי שבועות:

| שבוע | מטרה | קבצים | סטטוס |
|------|------|-------|-------|
| **1** | דאטה ימי נקי כמערך NumPy | `src/data/` , `src/inspect_data.py` | ✅ |
| **2** | חורים מלאכותיים + קו בסיס + RMSE/MAE | `src/masks.py`, `src/baseline.py`, `src/metrics.py`, `src/run_baseline.py` | ✅ |
| **3-4** | מודל U-Net + אימון ראשוני | `src/model.py`, `src/train.py` | בהמשך |
| **5** | הערכה: U-Net מול הבסיס על מפות held-out | `src/evaluate.py` | בהמשך |

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
```

## מסלול B — דאטה אמיתי מ-Copernicus
1. הרשמה (חינם): https://data.marine.copernicus.eu/register
2. `cp .env.example .env` ומלאי שם משתמש/סיסמה.
3. ודאי את `copernicus_dataset_id` המדויק לגלים בקטלוג, ואז:
```bash
python -m src.data.download
python -m src.data.load data/raw/<file>.nc
python -m src.run_baseline data/processed/<file>_64x64.npy
```

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

## פלטים
- `data/processed/*_64x64.npy` — מערך `(ימים, 64, 64)` מוכן למודל.
- `output/baseline_metrics.json` — RMSE/MAE של הבסיס (היעד שצריך לנצח).
- `output/baseline_example.png` — אמת / חורים / השלמה / שגיאה.

> **תוצאות דאטה סינתטי (רפרנס):** baseline RMSE ≈ 0.29 מ׳, MAE ≈ 0.20 מ׳ על 14 מפות.
> אלה מספרים על דאטה מדומה — יתעדכנו כשירוץ על נתוני Copernicus אמיתיים.

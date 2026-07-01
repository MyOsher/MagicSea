# MagicSea — Marine Map Gap-Filling POC

מטרת ה-POC: קוד פייתון שמקבל מפת ים עם חורים (ענן/נתון חסר) ומחזיר מפה שלמה בדיוק גבוה.

התוכנית לפי שבועות:

| שבוע | מטרה | קבצים רלוונטיים |
|------|------|-----------------|
| **1** | דאטה ימי נקי כמערך NumPy | `src/data/` , `src/inspect_data.py` |
| **2** | יצירת "חורים" (מסכות) על מפות שלמות | `src/masks.py` *(בהמשך)* |
| **3-4** | מודל U-Net + אימון ראשוני | `src/model.py`, `src/train.py` *(בהמשך)* |
| **5** | הערכה: RMSE/MAE על מפות שהמודל לא ראה | `src/evaluate.py` *(בהמשך)* |

---

## שבוע 1 — מה שכבר בנוי

### התקנה
```bash
pip install -r requirements.txt
```

### מסלול A — בלי חשבון Copernicus (מתחילים מיד)
דאטה סינתטי שמדמה מפות ים, בעל אותו מבנה כמו הדאטה האמיתי:
```bash
python -m src.data.synthetic                                   # יוצר data/raw/..._SYNTHETIC.nc
python -m src.data.load data/raw/mediterranean_sst_SYNTHETIC.nc # NetCDF -> NumPy 64x64
python -m src.inspect_data data/processed/mediterranean_sst_SYNTHETIC_64x64.npy
```
> יחידות (צלזיוס/קלווין) מזוהות אוטומטית ב-`load.py` — אין צורך לשנות דבר בין הדאטה הסינתטי לאמיתי.

### מסלול B — עם דאטה אמיתי מ-Copernicus
1. הרשמה (חינם): https://data.marine.copernicus.eu/register
2. `cp .env.example .env` ומלאי שם משתמש/סיסמה.
3. הורדה + עיבוד:
```bash
python -m src.data.download                                    # מוריד לפי config.yaml
python -m src.data.load data/raw/<file>.nc
python -m src.inspect_data data/processed/<file>_64x64.npy
```

### מה קורה מאחורי הקלעים
- `config.yaml` — מקום אחד לכל ההגדרות: אזור, טווח תאריכים, משתנה (ברירת מחדל SST), רזולוציית יעד (64×64).
- `download.py` — מוריד אזור קטן של הים התיכון לשבועיים בעזרת ה-toolbox הרשמי.
- `synthetic.py` — חלופה שמייצרת NetCDF זהה במבנה, כדי לא להיחסם.
- `load.py` — פותח NetCDF, ממיר יחידות, **מוריד רזולוציה ל-64×64** (טיפ המהירות המרכזי), שומר `.npy`. חורים נשמרים כ-`NaN` בכוונה.
- `inspect_data.py` — מדפיס סטטיסטיקות ושומר תצוגה מקדימה ל-`output/`.

**פלט שבוע 1:** קובץ `data/processed/*_64x64.npy` בצורת `(ימים, 64, 64)` + PNG תצוגה מקדימה — מוכן לשבוע 2.

# Aksiya narxlarini prognoz qilish paneli

Ushbu loyiha `Python` va `Streamlit` yordamida yaratilgan web ilova bo'lib,
aksiya narxlarini 10 yillik tarix asosida tahlil qiladi va uzoq muddatli
prognozlarni benchmark bilan solishtiradi.

## Asosiy imkoniyatlar

- Tayyor tickerlar: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN`, `GOOGL`
- Foydalanuvchi boshqa ticker ham kiritishi mumkin
- Oxirgi `10 yillik` ma'lumot `Yahoo Finance` orqali olinadi
- 4 ta o'rganadigan model ishlatiladi:
  - `Ridge Regression`
  - `Random Forest Regressor`
  - `Gradient Boosting Regressor`
  - `Extra Trees Regressor`
- Har bir model majburiy ravishda `Naive Baseline` bilan solishtiriladi
- Kunlik holdout test va uzoq muddat uchun rolling backtest mavjud
- Asosiy metrikalar:
  - `MAE`
  - `RMSE`
  - `MAPE`
  - `R2 Score`
- 3, 6 yoki 12 oylik prognoz
- Professional panel:
  - joriy narx
  - kunlik o'zgarish
  - 52 haftalik diapazon
  - asosiy candlestick chart
  - benchmark va modellar taqqoslangan forecast chart
  - rolling backtest reytingi
- Test natijalari va prognozlarni `CSV` formatida yuklab olish mumkin
- Internet bo'lmasa, ilova sintetik demo data bilan ishlashda davom etadi

## Fayllar tuzilmasi

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
`-- run_app.bat
```

## Ishga tushirish

### 1-usul: `run_app.bat`

Windows tizimida loyiha papkasiga kiring va `run_app.bat` faylini ishga tushiring.
U virtual muhitni yaratadi, kutubxonalarni o'rnatadi va ilovani ochadi.

### 2-usul: terminal orqali

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Brauzer manzili:

```text
http://localhost:8501
```

## Qisqa metodologiya

- Kunlik featurelar sifatida oxirgi 30 kunlik lag qiymatlar, rolling mean/std,
  return, momentum, EMA va vaqt indeksi ishlatiladi.
- Kunlik testda model keyingi kunlik narx o'zgarishini bashorat qiladi.
- Uzoq muddatli prognoz uchun haftalik returnlar bo'yicha direct multi-output
  yondashuv ishlatiladi.
- 3, 6 va 12 oylik prognozlar mos ravishda 13, 26 va 52 haftalik ufqda
  baholanadi.
- Tanlangan prognoz modeli bitta split bo'yicha emas, bir nechta rolling
  backtest oynalari bo'yicha aniqlanadi.
- Agar murakkab model `Naive Baseline`dan yaxshiroq chiqmasa, ilova buni yashirmaydi
  va benchmarkni eng ishonchli natija sifatida ko'rsatadi.

## Eslatma

- Aksiya narxlari juda shovqinli bo'lgani uchun soddaroq benchmarkni yengish ham
  qiyin bo'lishi mumkin.
- Model natijalari investitsion tavsiya emas; ular tarixiy ma'lumot asosidagi
  analitik ko'rsatkichlardir.

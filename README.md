# Aksiya narxlarini prognoz qilish paneli

Ushbu loyiha `Python` va `Streamlit` yordamida yaratilgan web ilova bo'lib,
aksiya narxlarini 5 yillik tarix asosida tahlil qiladi va 3 oylik prognozlarni
benchmark bilan solishtiradi.

## Asosiy imkoniyatlar

- Tayyor tickerlar: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN`, `GOOGL`
- Foydalanuvchi boshqa ticker ham kiritishi mumkin
- Standart tarix chuqurligi: `5 yil`
- 4 ta o'rganadigan model ishlatiladi:
  - `Ridge Regression`
  - `Random Forest Regressor`
  - `Gradient Boosting Regressor`
  - `Extra Trees Regressor`
- Qo'shimcha ravishda 2 ta yengil ansambl ishlatiladi:
  - `Mean Ensemble`
  - `Conservative Blend`
- Har bir model majburiy ravishda `Naive Baseline` bilan solishtiriladi
- Kunlik holdout test va uzoq muddat uchun rolling backtest mavjud
- Asosiy metrikalar:
  - `MAE`
  - `RMSE`
  - `MAPE`
  - `R2 Score`
- Standart prognoz ufqi: `3 oy`
- Asosiy panel:
  - joriy narx
  - to'liq tarixiy chart
  - benchmark va modellar taqqoslangan forecast chart
  - model test natijalari jadvali
  - train/test davri grafigi
- Internet bo'lmasa, ilova sintetik demo data bilan ishlashda davom etadi

## Fayllar tuzilmasi

```text
.
|-- app.py
|-- model_pipeline.py
|-- notebooks/
|   `-- stock_forecast_colab.ipynb
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
- Joriy ilovada 3 oylik prognoz 13 haftalik ufqda baholanadi.
- Tanlangan prognoz modeli bitta split bo'yicha emas, bir nechta rolling
  backtest oynalari bo'yicha aniqlanadi.
- `Mean Ensemble` mavjud 4 model prognozlarining o'rtachasi, `Conservative Blend`
  esa shu ansambl bilan benchmarkning ehtiyotkor aralashmasidir.
- Agar murakkab model `Naive Baseline`dan yaxshiroq chiqmasa, ilova buni yashirmaydi
  va benchmarkni eng ishonchli natija sifatida ko'rsatadi.

## Google Colab bilan ishlash

- Model logikasi endi alohida `model_pipeline.py` faylida saqlanadi.
- Streamlit ilova ham, Colab notebook ham ayni pipeline'dan foydalanadi.
- Tayyor notebook:
  `notebooks/stock_forecast_colab.ipynb`
- Colabda yangi model, feature yoki parametrni sinab ko'rib, yaxshi natija chiqsa
  o'sha o'zgarishni `model_pipeline.py`ga qo'shib GitHubga push qilish kifoya.

## Eslatma

- Aksiya narxlari juda shovqinli bo'lgani uchun soddaroq benchmarkni yengish ham
  qiyin bo'lishi mumkin.
- Model natijalari investitsion tavsiya emas; ular tarixiy ma'lumot asosidagi
  analitik ko'rsatkichlardir.

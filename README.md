# Aksiya narxlarini Machine Learning modellari orqali bashorat qilish

Ushbu loyiha `Python` va `Streamlit` yordamida yaratilgan web ilova bo‘lib, aksiya narxlarini 10 yillik tarix asosida tahlil qilish, 4 ta kuchli modelni solishtirish va kelajak narxlarini prognoz qilish imkonini beradi.

## Asosiy imkoniyatlar

- Tayyor tickerlar: `AAPL`, `MSFT`, `NVDA`, `TSLA`, `AMZN`, `GOOGL`
- Foydalanuvchi o‘zi boshqa ticker ham kiritishi mumkin
- Oxirgi `10 yillik` ma’lumot `Yahoo Finance` orqali olinadi
- Birinchi taxminan `9 yil` train, oxirgi `1 yil` test sifatida ishlatiladi
- Faqat 4 ta tanlangan model ishlatiladi:
  - `Linear Regression`
  - `Random Forest Regressor`
  - `Gradient Boosting Regressor`
  - `LSTM Neural Network`
- Baholash metrikalari:
  - `MAE`
  - `RMSE`
  - `MAPE`
  - `R² Score`
- 3, 6 yoki 12 oylik kelajak prognozi
- Zamonaviy UI: tepada ticker tanlash, real narx kartalari, real candlestick chart, 4 model uchun bitta forecast chart va test natijalari jadvali
- Test natijalari hamda prognozlarni `CSV` formatida yuklab olish imkoniyati
- Vaqt qatoriga mos `walk-forward` tekshiruv jadvali bilan model barqarorligi ham ko‘rsatiladi
- Internet bo‘lmasa, ilova sintetik demo data bilan ishlashda davom etadi

## Fayllar tuzilmasi

```text
.
├── app.py
├── requirements.txt
├── README.md
└── run_app.bat
```

## Ishga tushirish

### 1-usul: `run_app.bat` orqali

Windows tizimida loyiha papkasiga kiring va `run_app.bat` faylini ikki marta bosing.

U avtomatik ravishda:

1. `.venv` virtual muhitini yaratadi
2. Kerakli kutubxonalarni o‘rnatadi
3. Ilovani `Streamlit` orqali ishga tushiradi

### 2-usul: terminal orqali

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Brauzerda odatda quyidagi manzil ochiladi:

```text
http://localhost:8501
```

## Qisqa metodologiya

- Model uchun oxirgi 30 kunlik lag qiymatlar, 5/10/20/30 kunlik o‘rtacha qiymatlar, standart og‘ishlar, return, momentum, EMA va vaqt indeksi feature sifatida ishlatiladi.
- Modellar keyingi kunning to‘liq narxini emas, balki `kunlik narx o‘zgarishini` bashorat qiladi. Bu uzoq tarixiy data bilan barqarorroq ishlaydi.
- Test davrida har bir modelning bashorati haqiqiy narxlar bilan solishtiriladi.
- Bundan tashqari, model sifati vaqt tartibini saqlagan `walk-forward` tekshiruv bilan ham baholanadi.
- Eng yaxshi model `RMSE` bo‘yicha avtomatik tanlanadi.
- Kelajak prognozi uchun modellar barcha 10 yillik ma’lumot bilan qayta o‘qitiladi va uzoq muddat uchun `haftalik direct multi-horizon return forecasting` yondashuvi ishlatiladi.
- 3 oy uchun 13 hafta, 6 oy uchun 26 hafta, 12 oy uchun 52 hafta ketma-ket prognoz qilinadi; shu sabab model faqat birinchi kun harakatini ko‘chirib ketmaydi.

## Eslatma

- `LSTM Neural Network` modeli `TensorFlow / Keras` yordamida qurilgan recurrent neural network bo‘lib, vaqt qatorlari uchun mos keladi.
- `Linear Regression` tushunarli baseline sifatida, `Random Forest` va `Gradient Boosting` esa kuchli tree-based ML modellar sifatida ishlatiladi.
- Agar internet bo‘lmasa yoki `Yahoo Finance` javob bermasa, sahifada `Demo data ishlatilmoqda` degan ogohlantirish ko‘rinadi.
- Aksiya narxlari tabiatan shovqinli bo‘lgani uchun uzoq muddatli prognozlar har doim ehtiyotkor talqin qilinishi kerak.

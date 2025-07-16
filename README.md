F1 Podium Predictor

A simple and interactive [Streamlit](https://streamlit.io) web application that predicts whether a Formula 1 driver will finish on the podium based on race-specific input features.

---

Features

- Upload CSV files containing driver race data.
- Predict podium finish (`Yes/No`) using a pre-trained XGBoost model.
- Visualize model prediction confidence (%).
- View feature importance plot from the XGBoost model.

---

 Expected Input Format

The uploaded CSV should include the following columns:

- `grid` — Starting grid position.
- `age_at_race` — Driver’s age at time of race.
- `constructor_encoded` — Numeric encoding of the constructor/team.
- `year` — Year of the race.

---

 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kittu-t1/f1-podium-predictor.git
cd f1-podium-predictor

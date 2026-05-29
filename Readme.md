# Student Performance Predictor 🎓

A simple Streamlit web app that uses a PyTorch neural network to predict a student's performance index based on their study habits and lifestyle.

👉 **[Live App Link](https://wonderer99-student-performance.streamlit.app/)**

---

## What it does
* **Predicts Scores:** Enter data like study hours, sleep, and previous grades to instantly see a predicted performance score out of 100.
* **Gives Feedback:** Converts the score into a letter grade (A+ to D) and shows a dynamic success/warning card depending on how well the student did compared to the average.
* **Data Breakdown:** Includes an interactive Plotly bar chart at the bottom that reads directly from the dataset to show which factors actually impact performance the most.
* **Custom UI:** Custom dark-mode gradient theme using the Space Grotesk font.

---

## The Model Behind It
The app runs on a basic Feedforward Neural Network built with **PyTorch**:
* **Inputs:** 5 features (Hours Studied, Previous Scores, Extracurriculars, Sleep Hours, Sample Papers Practiced).
* **Architecture:** 5 → 64 (ReLU) → 32 (ReLU) → 1 (Output Score).
* **Preprocessing:** Inputs and outputs are scaled back and forth using `MinMaxScaler` / `StandardScaler` saved via `joblib`.

---

# Hyderabad House Price Prediction Model

A local Streamlit machine-learning project that estimates a property price from the Hyderabad housing CSV supplied for this project.

## What the app does

- Predicts one property-price point from **area, locality, bedrooms, new/resale status**, and optional amenity details.
- Shows dataset summaries, price-versus-area patterns, and locality medians.
- Explains the cross-validation method, the baseline comparison, and the limits of the data.
- Keeps “not specified” amenity values separate from “No.”

The supplied file has no bathroom or PIN-code field, so the app does not request or pretend to use those inputs. The CSV also has no collection date, which means this is not a live-market estimate.

## Project architecture

`Hyderabad.csv` → validation and duplicate removal → preprocessing pipeline → nested cross-validation and model comparison → fitted model and metrics → Streamlit UI.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture, data decisions, and tech stack.

## Windows setup and run

### Easiest: use the included launcher

1. Extract the ZIP into a normal folder, such as `C:\Users\YourName\Downloads\Hyderabad_Home_Price_Studio`.
2. Install Python 3 from [python.org](https://www.python.org/downloads/windows/) if it is not already installed. During installation, enable **Add Python to PATH**.
3. Double-click **`run_app.bat`**. The first run creates a private `.venv`, installs the listed packages, and opens the app in your browser. The trained model and evaluation results are included. First setup needs internet for package installation.
4. Leave the black terminal window open while you use the app. To stop it, focus that window and press **Ctrl+C**.

### Run from VS Code

1. In VS Code, use **File → Open Folder** and select the extracted `Hyderabad_Home_Price_Studio` folder.
2. Open **Terminal → New Terminal**. Confirm the terminal is in that project folder. If it is not, run:

   ```powershell
   cd "$HOME\Downloads\Hyderabad_Home_Price_Studio"
   ```

   Replace the folder location if you extracted it somewhere else.

3. Run these commands one at a time in the VS Code PowerShell terminal:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py
   ```

   If PowerShell blocks activation, the app can still be run using the launcher; alternatively, activate for this terminal session with:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

4. The browser opens the app. If it does not, copy the `Local URL` shown in the terminal (normally `http://localhost:8501`) into Chrome.
5. Stop the app with **Ctrl+C** in the terminal.

## Retrain after changing the CSV

Replace `Hyderabad.csv` with a CSV that retains the required columns, then run:

```powershell
python train_model.py
```

The training step writes `house_price_model.joblib` and `model_metrics.json` in this folder. A malformed file stops with an error that names the missing columns or invalid data rather than silently training a different model.

## Dataset notes

- The supplied CSV contains 2,518 source rows, 40 columns, and 243 locality labels.
- It includes `Price`, `Area`, `Location`, `No. of Bedrooms`, `Resale`, and amenity fields. It does not include bathrooms, PIN codes, or a collection date.
- The raw file includes 520 exact duplicate records; the training pipeline removes exact duplicates before evaluation.
- Area is identified as square feet in related dataset documentation. The source CSV does not include a currency code or define whether each price is an asking price or a completed transaction. The app displays ₹ based on the dataset’s Hyderabad, India context and explicitly labels that as an inference.
- The dataset is not live. Compare the result with current listings before making a decision.
- Data source references: [Hyderabad Kaggle dataset project](https://github.com/ShefaaSaied/Predicting-Housing-Prices-in-hyderabad-India) · [Grokking Machine Learning, Hyderabad example](https://dataforsolution.github.io/assets/books/Grokking%20Machine%20Learning.pdf). The exact download date and reuse license were not supplied with the CSV.

## Files

| File | Purpose |
|---|---|
| `app.py` | User interface, prediction flow, data exploration, methodology |
| `train_model.py` | Validation, preprocessing, nested evaluation, final model training |
| `Hyderabad.csv` | The CSV supplied for this project |
| `requirements.txt` | Python package dependencies |
| `run_app.bat` | Windows setup and launch helper |
| `ARCHITECTURE.md` | Architecture, user flow, model and stack overview |
| `test_project.py` | Automated checks for data validation and model behavior |
| `house_price_model.joblib` | Trained prediction and preprocessing pipeline |
| `model_metrics.json` | Cross-validation results and data audit |

The model artifact was trained from the included CSV. The dependency versions are pinned to the versions used for training and testing.

## Tech stack

**Languages:** Python, Markdown, HTML, CSS.  
**Libraries:** Streamlit, pandas, NumPy, scikit-learn, joblib, Plotly Express.  
**ML methods:** Ridge regression, Extra Trees regression, log-transformed target, one-hot categorical encoding, nested 5-fold cross-validation, MAE/RMSE/R², and a median baseline.

This is an educational ML application, not a formal property valuation service. Validation scores describe held-out rows from this dataset and do not provide a guaranteed price range for an individual home.

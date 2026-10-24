# SolarScope — Space Weather Forecasting Dashboard

## Files
- `app.py`: Streamlit dashboard
- `requirements.txt`: Python dependencies
- `kpdata.txt`: Place your downloaded GFZ combined-index data file here (not included)

## Run locally on your Mac

1. Put `app.py`, `requirements.txt`, and your downloaded `kpdata.txt` in the same folder.
2. Open Terminal in that folder.
3. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

6. Your browser should open the dashboard. If it does not, open the local URL printed in Terminal (usually http://localhost:8501).

You can also upload `kpdata.txt` using the dashboard's sidebar file uploader.

## Data assumptions
The app expects the GFZ combined `Kp, ap, Ap, SN, F10.7` whitespace-delimited format with at least 28 columns:
- columns 0–2: year, month, day
- columns 7–14: eight 3-hour Kp values
- columns 15–22: eight 3-hour ap values
- column 23: daily Ap
- column 24: sunspot number
- column 25: F10.7 flux
- column 26: centered 81-day F10.7 mean
- column 27: quality/status flag

The code treats Ap >= 400 as missing. Verify sentinel values and column definitions against the current GFZ format documentation before publishing results.

## Important interpretation note
The dashboard is for an academic project. A forecast based only on historical Ap does not predict local aurora visibility or provide operational storm warnings.

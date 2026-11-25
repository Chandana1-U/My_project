# MAJOR - SAR Ship Detector

Starter Flask structure for the SAR Ship Detection project.

## Current stage

This version contains:
- Flask backend
- Dashboard UI
- CSS separated from HTML
- JavaScript separated from HTML

The ML pipeline is intentionally not connected yet.

## Run

1. Open a terminal inside the MAJOR folder.
2. Create/activate a Python virtual environment if desired.
3. Install dependencies:

   pip install -r requirements.txt

4. Start the application:

   python app.py

5. Open:

   http://127.0.0.1:5000

## Next modules to add

- preprocessing/preprocess.py
- cfar/cfar_detector.py
- model/train.py
- model/predict.py
- model/best.pt
- evaluation/metrics.py
- evaluation/comparison.py
- uploads/
- results/
- dataset/

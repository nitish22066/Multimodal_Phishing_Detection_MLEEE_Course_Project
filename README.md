# Phishing URL Detection System

A multi-model ensemble system for detecting phishing URLs using network analysis, structural patterns, and content behavior.

> 📄 **For detailed technical documentation, see [REPORT.md](REPORT.md)**

## Setup Instructions

1. Create a virtual environment:
```bash
python3 -m venv .venv
```

2. Activate the virtual environment:
```bash
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Model Components

The system uses three different models for comprehensive phishing detection:

1. Network Analysis Model
   - Analyzes network characteristics and domain reputation
   - Located at: Models/model_network_analysis.pkl

2. Structure Analysis Model
   - Examines URL structure and patterns
   - Located at: Models/phishing_structure_analysis.pkl

3. Content Analysis Model
   - Analyzes webpage content and behavior
   - Located at: Models/phishing_page_content_analysis.pkl
     
4. Visual Analysis Model
   - Analyzes screenshot of website
   - Located at: Models/phishing_detector_screenshots.pkl
## Ensemble Meta-Learner (Advanced)

The system now includes an optional ensemble meta-learning layer that combines base models with:
- Graph Neural Network (GNN) feature encoding (optional, with MLP fallback)
- Partial Least Squares (PLS) dimensionality reduction (optional)
- Stacked meta-learner (Logistic Regression or Gradient Boosting)

### Using the Ensemble Layer

1. Train the ensemble model:
```bash
python Model_Training/ensemble_example.py
```

2. The ensemble model will be saved to `Models/ensemble_meta_learner.pkl`

3. Optional: Install PyTorch for enhanced GNN support:
```bash
pip install -r requirements-optional.txt
```

Note: The ensemble layer works without PyTorch, using sklearn-based fallbacks.

### CSV Evaluation (Kaggle-Style)

The app now includes a third tab for comprehensive CSV evaluation:

1. Navigate to the "Evaluate CSV (Kaggle-style)" tab in the app
2. Upload a CSV file with a `URL` column (and optionally a `label` column for metrics)
3. The system will:
   - Process each URL through the detection pipeline
   - Generate per-row predictions and probability scores
   - Calculate metrics if ground-truth labels are provided
   - Save results to `Models/csv_results/` with exact predictions

**CSV Format Example:**
```csv
URL,label,description
https://example.com,0,Legitimate site
http://suspicious-site.xyz,1,Phishing attempt
```

Results are saved with columns:
- Original columns from your CSV
- `__pred_label`: Predicted class (0=legitimate, 1=phishing)
- `__pred_score`: Confidence score (0.0 to 1.0)
- `__model_version`: Model version/timestamp

### Configuration

Edit `configuration.json` to customize ensemble behavior:
```json
{
  "ensemble_config": {
    "use_pls": false,
    "pls_components": 10,
    "use_gnn": false,
    "meta_learner_type": "logistic",
    "gnn_output_dim": 32,
    "k_neighbors": 5
  }
}
```

## Troubleshooting

1. If you see XGBoost warnings about serialized models, these can be safely ignored.

2. If you get ModuleNotFoundError:
   - Make sure you've activated the virtual environment
   - Try reinstalling the dependencies: `pip install -r requirements.txt`

3. If models fail to load:
   - Verify all .pkl files are in the correct locations
   - Check file permissions
   - Ensure you're using Python 3.8 or later

4. For SSL/Connection errors:
   - Check your internet connection
   - Verify the URL is accessible
   - Try with a different URL to confirm if it's a specific website issue

5. For ensemble layer issues:
   - Ensure base models are trained and exist in `Models/` directory
   - Run `python Model_Training/ensemble_example.py` to verify setup
   - Check `Models/csv_results/` for saved evaluation outputs

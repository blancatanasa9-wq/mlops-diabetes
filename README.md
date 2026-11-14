# MLOps End-to-End Pipeline & Prediction API: Diabetes Progression

An end-to-end MLOps pipeline covering data extraction, validation, preprocessing, model training, tracking with MLflow, automated CI/CD deployment via GitHub Actions, and production serving using FastAPI.

## 📋 Table of Contents

 1. [Project Goal](#-project-goal)

 2. [Dataset Used](#-dataset-used)

 3. [Repository Structure](#-repository-structure)

 4. [Installation & Dependencies](#-installation--dependencies)

 5. [Data Pipeline Execution](#-data-pipeline-execution)

 6. [Model Training & Registration](#-model-training--registration)

 7. [Champion Model Selection](#-champion-model-selection)

 8. [Running the API](#-running-the-api)

 9. [Testing the Endpoints](#-testing-the-endpoints)

10. [CI/CD Automation](#-cicd-automation)

11. [Error Handling](#-error-handling)

## 🎯 Project Goal

The objective of this project is to build a robust, production-ready MLOps pipeline for predicting disease progression (quantitative measure of diabetes one year post-baseline). The system automates data processing, tracks model experiments using MLflow, selects and registers the best model automatically, evaluates performance against strict quality gates, deploys an automated CI/CD integration workflow, and serves predictions via a high-performance FastAPI web service with strict validation and error management.

## 📊 Dataset Used

* **Source:** Standard Scikit-Learn / UCI Diabetes Regression Dataset.

* **Features (10 numerical variables):**

  * `age`: Age in years

  * `sex`: Sex

  * `bmi`: Body mass index

  * `bp`: Average blood pressure

  * `s1` through `s6`: Six blood serum measurement variables

* **Target:** Quantitative measure of disease progression one year after baseline.

## 🗂️ Repository Structure

```
mlops-diabetes-main/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── src/
│   ├── api/
│   │   ├── main.py                # FastAPI application entrypoint & endpoints
│   │   └── model.py               # Model loading & MLflow integration
│   ├── data/
│   │   ├── load_data.py           # Data extraction script
│   │   ├── validate_data.py       # Data validation checks
│   │   ├── preprocess.py          # Data cleaning & transformation
│   │   └── feature_engineering.py # Feature engineering pipeline
│   └── models/
│       ├── train.py               # Model training & MLflow tracking
│       ├── register_model.py      # Model registry integration
│       └── evaluate_model.py      # Quality gates & champion model assignment
├── requirements.txt               # Python project dependencies
└── README.md                      # Project documentation
```

## ⚙️ Installation & Dependencies

1. **Clone the repository:**

   ```bash
   git clone https://github.com/blancatanasa9-wq/mlops-diabetes.git
   cd mlops-diabetes
   ```

2. **Create and activate a Python virtual environment (Python 3.11):**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\Activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🔄 Data Pipeline Execution

To execute the pipeline manually in the correct sequence from raw data to processed features, run the following scripts in order:

```bash
# 1. Data Extraction
python src/data/load_data.py

# 2. Data Validation
python src/data/validate_data.py

# 3. Data Cleaning & Preprocessing
python src/data/preprocess.py

# 4. Feature Engineering
python src/data/feature_engineering.py
```

## 🤖 Model Training & Registration

Once the data is processed, train the models, track them with MLflow, register them, and evaluate quality gates using:

```bash
# 1. Train models (Linear Regression, Random Forest, etc.) & log to MLflow
python src/models/train.py

# 2. Register the best-performing model candidate
python src/models/register_model.py

# 3. Evaluate model against Quality Gates and assign the '@champion' alias
python src/models/evaluate_model.py
```

## 👑 Champion Model Selection & Quality Gates

The model selection process is fully automated via evaluation scripts (`evaluate_model.py`) and governed by strict **Quality Gates**:

1. **Candidate Evaluation:** All trained models are evaluated on a holdout test dataset.
2. **Quality Gate Threshold (`MIN_R2_THRESHOLD = 0.25`):** A candidate model is *only* promoted to **`@champion`** status if its coefficient of determination ($R^2$) meets or exceeds the defined threshold:
   * **$R^2 \ge 0.25$**
   * *Rationale:* This ensures that any model promoted to production must explain at least 25% of the variance in diabetes progression, filtering out underperforming or trivial baseline models.
3. **MLflow Alias Assignment:** Once a model passes this quality gate, it is automatically tagged with the MLflow alias **`@champion`**, ensuring the FastAPI serving layer always fetches the best available version dynamically.

## 🚀 Running the API

To launch the FastAPI prediction server locally using Uvicorn:

```bash
uvicorn src.api.main:app --reload
```

The server will start at `http://127.0.0.1:8000`. You can access the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

## 🧪 Testing the Endpoints

You can test the core endpoints directly via Swagger UI or using `curl`:

1. **Health Check (`GET /health`):**

   * Verifies service operational status.

   * **Response:** `{"status": "ok"}`

2. **Model Info (`GET /model-info`):**
   * **Purpose:** Shows which model is currently being served in production, providing full traceability back to MLflow.
   * **Expected Response:** Returns active champion model metadata including name, version, alias, run ID, and tracking URI.
   * **Response:**
     ```json
     {
       "registered_model_name": "DiabetesBestRegressor",
       "alias": "champion",
       "model_version": "1",
       "run_id": "3303a1a9273f495391b08e8805f01e72",
       "tracking_uri": "sqlite:///mlflow.db"
     }
     ```

3. **Prediction (`POST /predict`):**
   * **Purpose:** Sends standardized patient features to get a diabetes progression prediction from the champion model.
   * **Example Payload:**
     ```json
     {
       "age": 0.038,
       "sex": 0.050,
       "bmi": 0.061,
       "bp": 0.021,
       "s1": -0.044,
       "s2": -0.034,
       "s3": -0.043,
       "s4": -0.0025,
       "s5": 0.019,
       "s6": -0.017
     }
     ```

   * **Successful Response (`200 OK`):**

     ```json
     {
       "prediction": 153.0008,
       "model_name": "DiabetesBestRegressor",
       "model_version": "1"
     }
     ```

## 🔄 CI/CD Automation

The project includes an automated Continuous Integration pipeline located at `.github/workflows/ci.yml`.

* **Triggers:** Automatically runs on every `push` to the `main` or `master` branches.

* **Execution Flow:** Sets up Python 3.11, installs dependencies, and runs the complete MLOps pipeline sequentially (`load_data` → `validate_data` → `preprocess` → `feature_engineering` → `train` → `register` → `evaluate`).

## 🛡️ Error Handling

The API is hardened against faulty inputs or infrastructure failures with standard HTTP status codes:

* **`200 OK`**: Successful prediction returned.

* **`422 Unprocessable Entity`**: Automatically managed by Pydantic when input data is missing, malformed, or has incorrect data types.

* **`503 Service Unavailable`**: Triggered if the MLflow champion model fails to load into memory or the prediction service is unavailable.

* **`500 Internal Server Error`**: Catches unexpected runtime/inference exceptions during execution.

# Step 01: Foundation & Data (Phase 1)

Time: 4–6 hours

Sub-steps:
- 1.1 Environment Setup (1–2 hours)
- 1.2 Data Acquisition (1 hour)
- 1.3 Exploratory Data Analysis (2–3 hours)

---

## 1.1 Environment Setup (1–2 hours)

Tasks
- Create Python virtual environment
- Install all required dependencies from requirements.txt
- Install and start Docker Desktop
- Initialize Git repository
- Create full project directory structure

Deliverables
- Working Python virtual environment
- Complete repository structure with all folders
- Docker running and verified
- Git initialized with initial commit

How to Validate
- Run python --version — should show 3.9+
- Run pip list — should show all packages (pandas, mlflow, bentoml, etc.)
- Run docker --version and docker-compose --version — both should work
- Run docker run hello-world — should succeed
- Check folder structure — all folders from Repository Structure should exist
- Run git status — should show clean working directory

---

## 1.2 Data Acquisition (1 hour)

Tasks
- Download NASA Turbofan dataset from Kaggle
- Extract files to data/raw/
- Verify file integrity and structure

Dataset Files Needed
- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt

Deliverables
- Raw data files in data/raw/
- Simple data loading test script

How to Validate
- Check data/raw/ folder contains all 3 .txt files
- Create quick Python script to load training data with pandas (space-separated)
- Verify training data has ~20,000+ rows and 26 columns (after dropping NaN columns)
- Verify unique engine units (unit_id column) = 100
- Print basic stats: number of rows, columns, unique engines

---

## 1.3 Exploratory Data Analysis (2–3 hours)

Tasks
- Create comprehensive EDA notebook
- Analyze sensor readings and their distributions
- Identify degradation patterns in sensors over cycles
- Calculate failure cycle statistics per engine
- Identify which sensors vary (std > 0) vs constant sensors

Deliverables
- notebooks/01_eda.ipynb with visualizations
- Summary document of key findings
- List of useful sensors (those that show variation)

How to Validate
- Jupyter notebook runs without errors
- Notebook answers these questions:
  - How many engine units total? (Should be 100)
  - Average cycles per engine? (Should be ~200)
  - Which sensors show meaningful variation? (About 14 out of 21)
  - Are there missing values? (Should be none after cleaning)
  - What's the RUL distribution?
- Create at least 3–4 visualizations (sensor trends, cycle distributions, correlation matrix)
- Save summary findings to text/markdown file

# Tetuan City Power Consumption Analysis

## Overview

This project analyzes the power consumption patterns of Tetuan City using machine learning techniques. The analysis includes exploratory data analysis (EDA), data preprocessing, and predictive modeling to understand electricity usage patterns across different zones and time periods.

## Dataset

The dataset contains power consumption data from Tetuan City, Morocco, collected over a period of time. It includes:

- **DateTime**: Timestamp of the readings
- **Temperature**: Ambient temperature
- **Humidity**: Relative humidity
- **Wind Speed**: Wind speed measurements
- **Diffuse Flows** and **General Diffuse Flows**: Solar irradiance measurements
- **Zone 1, 2, 3 Power Consumption**: Electricity consumption in kilowatts for three different zones

**Source**: UCI Machine Learning Repository

## Project Structure

```
├── data/
│   ├── Tetuan City power consumption.csv    # Main dataset
│   └── sets_info.csv                        # Dataset information
├── results/                                 # Generated plots and outputs
│   ├── missing_values_heatmap.png
│   ├── correlation_heatmap.png
│   ├── target_distribution.png
│   └── Training Accuracy.txt
├── draft.ipynb                              # Data ingestion and initial analysis
├── EDA_grp2.ipynb                          # Exploratory Data Analysis notebook
├── model.py                                 # Machine learning model training
├── requirements.txt                         # Python dependencies
├── .github/workflows/train.yml             # GitHub Actions CI/CD pipeline
└── README.md                               # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd applied-project-cl07_g02
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Exploratory Data Analysis

Run the EDA notebook to explore the data:
```bash
jupyter notebook EDA_grp2.ipynb
```

Or run the draft notebook for data ingestion pipeline:
```bash
jupyter notebook draft.ipynb
```

### Model Training

Run the machine learning model:
```bash
python model.py
```

This will perform data preprocessing, train models, and save results to the `results/` directory.

### GitHub Actions

The project includes a GitHub Actions workflow that automatically runs model training on pushes to the main branch. Results are uploaded as artifacts that can be downloaded from the Actions tab.

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- scipy: Scientific computing
- scikit-learn: Machine learning algorithms

## Results

The analysis generates various visualizations and metrics:

- Missing values heatmap
- Correlation analysis
- Feature distributions
- Model performance metrics
- Training accuracy reports

All outputs are saved in the `results/` directory.

## Contributors

- Group 2 (COS40007 Applied Project)
- Members: [Add team member names if available]

## License

This project is part of COS40007 Applied Project coursework.</content>
<parameter name="filePath">/Users/jamessluong_1/COS40007/applied-project-cl07_g02/README.md
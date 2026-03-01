# F1 Performance Analyzer & Predictor

## Overview
This project is an end-to-end data analysis and machine learning pipeline for predicting Formula 1 race outcomes and analyzing historical F1 data from 1950 to 2024. Using a rich dataset of races, drivers, constructors, circuits, qualifying, and lap times, the system uncovers trends, visualizes performance metrics, and trains a Random Forest Regression model to predict driver finishing positions.

## Features & Capabilities

### 1. Data Cleaning & Preprocessing
- Automatically handles missing indicator values (`\N`), replacing them with proper NaNs.
- Generates detailed missing value reports across 14 different datasets before and after cleaning.
- Imputes missing numerical values with their median and categorical values with the mode.
- De-duplicates datasets and formats dates for accurate temporal analysis.

### 2. Exploratory Data Analysis (EDA) & Visualizations
- **Historical Trends:** Visualizes F1 races added per year and the evolution of average lap times over the decades (1950-2024).
- **Eras of Dominance:** Tracks driver and constructor wins year-over-year.
- **Network Analysis:** Uses `networkx` to map driver transitions and movements between different constructors/teams visually.
- **Correlation Heatmaps:** Evaluates the relationship between various engineered features and race outcomes.

### 3. Feature Engineering
Extracts insightful racing features out of raw metrics to significantly enhance model predictive power:
- **Driver Consistency:** Computes the average race finishing position and average qualifying grid position per driver.
- **Team Momentum:** Calculates the rolling average of constructor points over the last 5 races.
- **Reliability Metric:** Computes the DNF (Did Not Finish) rates for constructors based on historical failure data.
- **Track Dynamics:** Measures overtake difficulty per race by comparing starting grid positions versus final finishing positions.

### 4. Machine Learning Predictive Model
- **Algorithm:** Trains a `RandomForestRegressor` to predict a driver's final finishing position (`positionOrder`).
- **Features Used:** Grid starting position, constructor rolling points, historical driver consistency, team DNF rates, and circuit overtake difficulty.
- **Model Evaluation:** The regressor is rigorously evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score, and 5-fold Cross-Validation to validate generalization.

### 5. In-Depth Racing Insights & Analysis
- **Qualifying vs. Race Day:** Identifies which drivers consistently gain or lose the most positions relative to their starting grid.
- **Pit Stop Strategy:** Evaluates the impact of pit stop frequency and average pit time on final race standings.
- **Head-to-Head Rivalries:** Pinpoints dominant drivers in direct head-to-head intra-race matchups.
- **Championship Statistics:** Analyzes the probability of consecutive championship retentions and trends regarding the average age of F1 champions through the decades.
- **Future Outlook (2025):** Projects the top drivers likely to compete for the 2025 championship and identifies mathematically struggling teams based on aggregated trends.

## Dataset Structure
The project pulls from multiple granular CSV files to form a complete picture of a race weekend:
- **Core Entities:** `circuits.csv`, `constructors.csv`, `drivers.csv`
- **Events & Status:** `races.csv`, `seasons.csv`, `status.csv`
- **Race Outcomes:** `results.csv`, `sprint_results.csv`, `qualifying.csv`
- **In-Race Metrics:** `lap_times.csv`, `pit_stops.csv`
- **Championship Standings:** `constructor_results.csv`, `constructor_standings.csv`, `driver_standings.csv`

## Setup & Installation

### Requirements
Ensure you have Python 3.x installed. The project relies on the following standard data science packages:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `networkx`

You can install all required dependencies easily using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn networkx
```

### Running the Project
1. Clone or download this repository.
2. Ensure all 14 dataset `.csv` files are located in the same root directory as the Python script.
3. Execute the main script from your terminal:
```bash
python mainCode.py
```

*Note: The script will sequentially output DataFrame summaries, model evaluation metrics to the console, and open several `matplotlib` and `seaborn` visualization windows. Close each graphical window to allow the script to proceed to the next analysis step.*

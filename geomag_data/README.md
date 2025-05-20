# BBC Microbit Magnetometer Data Analysis

This project analyzes magnetometer data collected from BBC Microbit devices to identify patterns related to geomagnetic storms and potential magnetic pole shifts.

## Overview

The analysis pipeline processes magnetometer data collected from multiple BBC Microbit devices placed at fixed locations. The data includes three-axis magnetic field readings (X, Y, Z) and temperature measurements.

The analysis focuses on:

1. **Basic Magnetometer Analysis**: Time series visualization, spectral analysis, and anomaly detection
2. **Geomagnetic Storm Correlation**: Comparing readings with known geomagnetic storm events
3. **Magnetic Pole Shift Analysis**: Analyzing trends in magnetic field orientation over time

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your magnetometer data is in the root directory as `collection_data.json`
2. Run the analysis pipeline:
   ```
   python run_analysis.py
   ```
3. Results will be saved in the `analysis_results` directory

## Data Format

The expected format for `collection_data.json` is a MongoDB export containing records with:

- `timestamp`: Time of measurement
- `device_id`: Unique identifier for the device
- `magnetometer`: Object containing `x`, `y`, and `z` components
- `temperature`: Temperature reading (optional)

Example:
```json
[
  {
    "timestamp": "2023-04-01T12:00:00Z",
    "device_id": "microbit-01",
    "magnetometer": {
      "x": 123.45,
      "y": -67.89,
      "z": 45.67
    },
    "temperature": 22.5
  },
  ...
]
```

## Analysis Components

### 1. Basic Magnetometer Analysis (`analyze_magnetometer_data.py`)

- Time series visualization of X, Y, Z components and total intensity
- Spectral analysis to identify periodic patterns
- Anomaly detection using statistical methods
- Temperature correlation analysis

### 2. Geomagnetic Storm Correlation (`compare_with_geomagnetic_storms.py`)

- Comparison with known geomagnetic storm events
- Statistical analysis of magnetic field behavior during storm periods
- Visualization of storm impacts on local measurements

### 3. Magnetic Pole Shift Analysis (`analyze_pole_shifts.py`)

- Calculation of magnetic field orientation (declination and inclination)
- Trend analysis to detect systematic changes over time
- Comparison with published rates of magnetic pole movement

## Output

The analysis generates various visualizations and data files in the `analysis_results` directory:

- Time series plots of magnetic components
- Spectral analysis plots
- Correlation heatmaps
- Storm period comparison plots
- Magnetic orientation trend plots
- CSV files with processed data and statistics

## Scientific Context

Earth's magnetic field is constantly changing, with the magnetic poles moving at rates of approximately 55-60 km per year in recent decades. Additionally, geomagnetic storms caused by solar activity can temporarily disturb the local magnetic field.

This analysis attempts to detect these phenomena using consumer-grade magnetometers, which may provide insights into:

- Local magnetic anomalies
- Effects of geomagnetic storms at specific locations
- Long-term trends in magnetic field orientation

## Limitations

- BBC Microbit magnetometers have limited precision and may be affected by local interference
- Fixed device locations limit the spatial analysis possibilities
- Temperature and other environmental factors may influence readings
- Short observation periods may not capture long-term trends effectively 
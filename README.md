# Temperature Controller Testing

This repository contains scripts for analyzing the temperature control performance of a Lakeshore 336 temperature controller, including step response testing and temperature sweep analysis.

## Scripts

### `run_temperature_step.py`
A comprehensive step response test script that:
- Performs controlled temperature step tests (e.g., 3.5K → 6.0K)
- Records baseline temperature before applying step
- Analyzes controller response characteristics
- Generates detailed plots and CSV data files
- Uses configuration files for easy parameter adjustment

## Usage

### Temperature Step Response Test:
```bash
python scripts/run_temperature_step.py
```

This script will:
1. Connect to the Lakeshore 336 controller
2. Set initial temperature and wait for stability
3. Record baseline temperature data
4. Apply temperature step to target setpoint
5. Record response data and analyze results
6. Generate plots and save all data to `step_response_results/`

## Configuration

Edit `configs/step_response_config.yaml` to customize:

```yaml
instrument:
  port: "GPIB0::12::INSTR"
  output_channel: 1

step_test:
  start_temp: 3.5        # Starting temperature (K)
  target_temp: 6.0       # Target temperature (K)
  recording_interval: 0.5 # Data recording interval (s)
  baseline_time: 30      # Baseline recording time (s)
  step_hold_time: 30     # Response recording time (s)

stability:
  tolerance: 0.1         # Stability tolerance (K)
  required_stable_readings: 3
  max_wait_initial: 300  # Max wait for initial stability (s)
```

## Output Structure

All test results are saved to the `step_response_results/` directory:

```
step_response_results/
├── step_response.log                    # Detailed test log
├── step_response_YYYYMMDD_HHMMSS.csv   # Raw data with timestamps
└── step_response_YYYYMMDD_HHMMSS.png   # Response plots
```

### Output Files:
- **Log files**: Real-time test progress and analysis results
- **CSV files**: Complete time-series data with temperature, setpoint, and errors
- **Plot files**: Dual-panel plots showing temperature response and tracking error

## Features

- **Configuration-driven**: All parameters controlled via YAML config files
- **Robust logging**: Detailed console and file logging for debugging
- **Automatic stability checking**: Waits for temperature stabilization
- **Error handling**: Setpoint retry logic and connection verification
- **Real-time monitoring**: Progress updates during test execution
- **Data analysis**: Automatic calculation of response characteristics
- **Professional plots**: Publication-ready plots with error analysis

## Requirements

- Python 3.8+
- qnnpy library
- pandas, numpy, pyyaml, matplotlib
- Lakeshore 336 connected via GPIB
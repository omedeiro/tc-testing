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

### `run_temperature_ramp.py`
A temperature ramp test script that demonstrates the controller's ramping capabilities:
- Performs linear temperature ramps using built-in controller functions
- Ramps from start temperature to target, holds, then ramps back down
- Analyzes ramp performance and tracking accuracy
- Configurable ramp rates and hold times
- Generates phase-colored plots showing ramp progression

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

### Temperature Ramp Test:
```bash
python scripts/run_temperature_ramp.py
```

This script will:
1. Connect to the Lakeshore 336 controller
2. Set initial temperature and wait for stability
3. Enable ramping and ramp to target temperature
4. Hold at target temperature for specified time
5. Ramp back down to starting temperature
6. Analyze ramp performance and save results to `ramp_test_results/`

## Configuration

### Step Response Test
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
```

### Ramp Test
Edit `configs/ramp_test_config.yaml` to customize:

```yaml
instrument:
  port: "GPIB0::12::INSTR"
  output_channel: 1

ramp_test:
  start_temp: 3.5        # Starting temperature (K)
  target_temp: 6.0       # Target temperature (K)
  ramp_rate: 1.0         # Ramp rate (K/min)
  recording_interval: 1.0 # Data recording interval (s)
  hold_time: 60          # Hold time at target (s)
```

Available ramp rate presets:
- `slow_ramp`: 0.5 K/min, 2 min hold
- `medium_ramp`: 1.0 K/min, 1 min hold  
- `fast_ramp`: 2.0 K/min, 30s hold

## Output Structure

All test results are saved to dedicated output directories:

### Step Response Test
```
step_response_results/
├── step_response.log                    # Detailed test log
├── step_response_YYYYMMDD_HHMMSS.csv   # Raw data with timestamps
└── step_response_YYYYMMDD_HHMMSS.png   # Response plots
```

### Ramp Test
```
ramp_test_results/
├── temperature_ramp.log                # Detailed test log
├── ramp_test_YYYYMMDD_HHMMSS.csv      # Raw data with timestamps
└── ramp_test_YYYYMMDD_HHMMSS.png      # Ramp plots with phase coloring
```

### Output Files:
- **Log files**: Real-time test progress and analysis results
- **CSV files**: Complete time-series data with temperature, setpoint, errors, and test phases
- **Plot files**: Professional plots showing temperature tracking and error analysis

## Features

- **Multiple test types**: Step response and linear ramp tests
- **Built-in controller functions**: Uses Lakeshore 336's native ramping capabilities
- **Configuration-driven**: All parameters controlled via YAML config files
- **Robust logging**: Detailed console and file logging for debugging
- **Automatic stability checking**: Waits for temperature stabilization
- **Error handling**: Setpoint retry logic and connection verification
- **Real-time monitoring**: Progress updates during test execution
- **Comprehensive analysis**: Automatic calculation of response and ramp characteristics
- **Professional plots**: Publication-ready plots with phase identification and error analysis

## Requirements

- Python 3.8+
- qnnpy library
- pandas, numpy, pyyaml, matplotlib
- Lakeshore 336 connected via GPIB
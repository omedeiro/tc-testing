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

### `run_measure_resistance_ramp.py`
A combined temperature ramp and resistance measurement script that:
- Simultaneously controls temperature and measures electrical resistance
- Uses Keithley 2400 sourcemeter for 2-wire resistance measurements
- Sources constant current (1 µA) and measures voltage to calculate resistance
- Records temperature, voltage, and resistance data throughout the ramp
- Analyzes temperature coefficient of resistance
- Generates comprehensive plots including resistance vs temperature
- Automatically enables/disables sourcemeter output with the temperature ramp

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

### Resistance Measurement Ramp Test:
```bash
python scripts/run_measure_resistance_ramp.py
```

This script will:
1. Connect to the Lakeshore 336 controller and Keithley 2400 sourcemeter
2. Set initial temperature and wait for stability
3. Enable resistance measurement and set constant current
4. Ramp to target temperature, hold, then ramp back down
5. Record temperature, voltage, and resistance data
6. Analyze temperature coefficient of resistance and save results to `resistance_ramp_results/`

## Configuration

### Temperature Step Response Test
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

### Resistance Measurement Ramp Test
Edit `configs/resistance_ramp_config.yaml` to customize:

```yaml
instrument:
  port: "GPIB0::12::INSTR"
  output_channel: 1

sourcemeter:
  port: "GPIB0::24::INSTR"
  current_level: 1.0e-6    # Source current for resistance measurement (A)

ramp_test:
  start_temp: 3.5          # Starting temperature (K)
  target_temp: 6.0         # Target temperature (K)
  ramp_rate: 1.0           # Ramp rate (K/min)
  recording_interval: 1.0  # Data recording interval (s)
  hold_time: 60            # Hold time at target (s)

pid:
  P: 100                   # Proportional gain
  I: 25                    # Integral gain
  D: 15                    # Derivative gain
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

### Resistance Measurement Ramp Test
```
resistance_ramp_results/
├── resistance_ramp.log                 # Detailed test log
├── resistance_ramp_YYYYMMDD_HHMMSS.csv # Raw data with temperature, voltage, resistance
├── resistance_ramp_YYYYMMDD_HHMMSS.png # Time-series plots (temp, resistance, error)
└── resistance_vs_temp_YYYYMMDD_HHMMSS.png # Resistance vs temperature plot with linear fit
```

### Output Files:
- **Log files**: Real-time test progress and analysis results
- **CSV files**: Complete time-series data with temperature, setpoint, errors, and test phases
- **Plot files**: Professional plots showing temperature tracking and error analysis

## Features

- **Multiple test types**: Step response, linear ramp tests, and resistance measurement
- **Simultaneous measurements**: Combined temperature control and electrical resistance measurements
- **Built-in controller functions**: Uses Lakeshore 336's native ramping capabilities
- **2-wire resistance measurement**: Configurable current source with voltage measurement
- **Temperature coefficient analysis**: Automatic calculation of dR/dT
- **Configuration-driven**: All parameters controlled via YAML config files
- **Robust logging**: Detailed console and file logging for debugging
- **Automatic stability checking**: Waits for temperature stabilization
- **Instrument coordination**: Synchronized control of temperature controller and sourcemeter
- **Error handling**: Setpoint retry logic and connection verification
- **Real-time monitoring**: Progress updates during test execution
- **Comprehensive analysis**: Automatic calculation of response and ramp characteristics
- **Professional plots**: Publication-ready plots with phase identification and error analysis

## Requirements

- Python 3.8+
- qnnpy library
- pandas, numpy, pyyaml, matplotlib
- Lakeshore 336 connected via GPIB
- Keithley 2400 sourcemeter (for resistance measurement script)
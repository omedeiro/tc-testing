# Keithley 2400 API Reference for Resistance Measurements

Based on the actual qnnpy.instruments.keithley_2400.Keithley2400 class, here are the correct method names and their usage:

## Setup Methods
- `setup_2W_source_I_read_V()` - Configure for 2-wire current source, voltage measurement
- `setup_2W_source_V_read_I()` - Configure for 2-wire voltage source, current measurement  
- `setup_4W_source_I_read_V()` - Configure for 4-wire current source, voltage measurement

## Current/Voltage Setting
- `set_current(current=0.0)` - Set source current in Amperes
- `set_voltage(voltage=0.0)` - Set source voltage in Volts

## Compliance Setting
- `set_compliance_v(voltage)` - Set voltage compliance limit
- `set_compliance_i(current)` - Set current compliance limit

## Measurement Methods
- `read_voltage()` - Read voltage measurement
- `read_current()` - Read current measurement
- `read_resistance()` - Read resistance measurement
- `read_voltage_and_current()` - Read both voltage and current

## Output Control
- `set_output(state)` - Enable (True) or disable (False) output

## Other Methods
- `reset()` - Reset instrument to default state
- `set_measurement_time(time)` - Set integration time for measurements
- `query(command)` - Send SCPI query and return response
- `write(command)` - Send SCPI command

## Example Usage for 2-Wire Resistance Measurement:
```python
from qnnpy.instruments.keithley_2400 import Keithley2400

# Initialize
keithley = Keithley2400('GPIB0::24::INSTR')

# Configure for 2-wire current source, voltage measurement
keithley.setup_2W_source_I_read_V()

# Set current and compliance
keithley.set_current(1e-6)  # 1 µA
keithley.set_compliance_v(10.0)  # 10V compliance

# Enable output
keithley.set_output(True)

# Measure voltage and calculate resistance
voltage = keithley.read_voltage()
resistance = voltage / 1e-6  # R = V/I

# Disable output
keithley.set_output(False)
```

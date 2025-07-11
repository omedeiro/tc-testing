"""
Shared utilities for temperature controller testing scripts.

This module contains common functions used by both step response and ramp test scripts
for the Lakeshore 336 temperature controller.
"""

import time
import logging
import yaml
from qnnpy.instruments.lakeshore336 import Lakeshore336

def load_config(config_file):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_file} not found. Using defaults.")
        return None


def setup_controller(inst: Lakeshore336, output, config):
    """Configure the temperature controller for testing."""
    logging.info("Setting up temperature controller")
    
    # Get temperature control settings
    temp_control_config = config.get('temperature_control', {}) if config else {}
    heater_range = temp_control_config.get('heater_range', 2)  # Default to medium range
    manual_output = temp_control_config.get('manual_output', 0)  # 0 = auto PID mode
    max_heater_output = temp_control_config.get('max_heater_output', 100)
    
    # Enable PID control with specified heater range
    logging.info(f"Setting heater range to {heater_range} (1=low, 2=medium, 3=high)")
    inst.set_output(output=output, mode=1, input_sensor=1, enable=1)
    inst.set_range(output=output, range_set=heater_range)
    
    # Set manual output if specified (0 = auto PID mode)
    if manual_output > 0:
        logging.info(f"Setting manual heater output to {manual_output}%")
        inst.set_manual_output(output=output, value=manual_output)
    else:
        logging.info("Using automatic PID control mode")
    
    # Configure PID parameters
    pid_config = config.get('pid', {}) if config else {}
    p_value = pid_config.get('P', 50)   # Proportional gain (reduced default)
    i_value = pid_config.get('I', 50)   # Integral gain (increased default)
    d_value = pid_config.get('D', 5)    # Derivative gain (reduced default)
    
    logging.info(f"Setting PID parameters: P={p_value}, I={i_value}, D={d_value}")
    inst.set_pid(output=output, P=p_value, I=i_value, D=d_value)
    
    # Verify PID settings
    time.sleep(0.5)
    current_pid = inst.get_pid(output=output)
    pid_values = current_pid.strip().split(',')
    
    if len(pid_values) == 3:
        actual_p = float(pid_values[0])
        actual_i = float(pid_values[1])
        actual_d = float(pid_values[2])
        
        logging.info(f"PID verification: P={actual_p}, I={actual_i}, D={actual_d}")
        
        # Check if values match (within tolerance)
        if (abs(actual_p - p_value) < 0.1 and 
            abs(actual_i - i_value) < 0.1 and 
            abs(actual_d - d_value) < 0.1):
            logging.info("✓ PID settings verified successfully")
        else:
            logging.warning(f"⚠ PID mismatch! Expected: P={p_value}, I={i_value}, D={d_value}")
    else:
        logging.warning(f"⚠ Could not parse PID response: {current_pid}")
    
    # Log final heater configuration
    logging.info(f"Heater configuration: Range={heater_range}, Max Output={max_heater_output}%")
    
    return True


def set_setpoint_with_retry(inst: Lakeshore336, setpoint, output, max_attempts=5):
    """Set setpoint with verification and retry logic."""
    for attempt in range(max_attempts):
        inst.set_setpoint(setpoint=setpoint, output=output)
        time.sleep(0.5)
        
        current_setpoint = float(inst.get_setpoint(output=output))
        if abs(current_setpoint - setpoint) < 0.05:
            return True
        
        if attempt < max_attempts - 1:
            time.sleep(0.5)
    
    raise RuntimeError(f"Failed to set setpoint to {setpoint}K after {max_attempts} attempts")


def set_setpoint_for_ramp(inst: Lakeshore336, setpoint, output):
    """Set setpoint for ramping - no retry logic needed as setpoint changes gradually."""
    inst.set_setpoint(setpoint=setpoint, output=output)
    time.sleep(0.1)  # Brief pause to ensure command is processed
    return True


def wait_for_stability(inst: Lakeshore336, target_temp, tolerance=0.1, max_wait=300, check_interval=1):
    """Wait for temperature to stabilize within tolerance of target."""
    start_time = time.time()
    stable_count = 0
    required_stable_readings = 5
    
    while time.time() - start_time < max_wait:
        current_temp = float(inst.read_temp(channel="A"))
        
        if abs(current_temp - target_temp) <= tolerance:
            stable_count += 1
            if stable_count >= required_stable_readings:
                return True, current_temp, time.time() - start_time
        else:
            stable_count = 0
            
        time.sleep(check_interval)
    
    final_temp = float(inst.read_temp(channel="A"))
    return False, final_temp, max_wait


def configure_ramp(inst: Lakeshore336, output, ramp_rate, enable=True, config=None):
    """Configure the temperature ramp settings with optional rate limiting."""
    
    # Apply ramp rate limit if specified in config
    if config:
        temp_control_config = config.get('temperature_control', {})
        ramp_rate_limit = temp_control_config.get('ramp_rate_limit', None)
        
        if ramp_rate_limit and ramp_rate > ramp_rate_limit:
            logging.warning(f"Ramp rate {ramp_rate:.2f} K/min exceeds limit {ramp_rate_limit:.2f} K/min, limiting to {ramp_rate_limit:.2f} K/min")
            ramp_rate = ramp_rate_limit
    
    logging.info(f"Configuring ramp: rate={ramp_rate:.2f} K/min, enabled={enable}")
    inst.set_ramp(output=output, on=1 if enable else 0, rate_value=ramp_rate)
    
    # Verify ramp configuration
    time.sleep(0.5)
    ramp_status = inst.get_ramp(output=output)
    logging.info(f"Ramp status: {ramp_status}")
    
    return ramp_rate  # Return the actual ramp rate used


def shutdown_heater(inst: Lakeshore336, output):
    """Safely shut down the heater to allow return to base temperature."""
    logging.info("Shutting down heater to allow return to base temperature")
    
    # Disable ramping first
    inst.set_ramp(output=output, on=0, rate_value=1.0)
    
    # Turn off the heater output
    inst.set_output(output=output, mode=0, input_sensor=1, enable=0)
    
    # Set range to off
    inst.set_range(output=output, range_set=0)
    
    logging.info("Heater shutdown complete - system will cool to base temperature")

import qnnpy.functions.functions as qf
import qnnpy.instruments.lakeshore336 as ctl
import time
import numpy as np
import pandas as pd
from datetime import datetime
import os
import logging
import matplotlib.pyplot as plt
import yaml

# Create output directory
output_dir = "step_response_results"
os.makedirs(output_dir, exist_ok=True)

# Configure logging
log_filename = os.path.join(output_dir, 'step_response.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)

def load_config(config_file="configs/step_response_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_file} not found. Using defaults.")
        return None

def setup_controller(inst, output, config):
    """Configure the temperature controller for step response."""
    logging.info("Setting up temperature controller")
    
    # Enable PID control
    inst.set_output(output=output, mode=1, input_sensor=1, enable=1)
    inst.set_range(output=output, range_set=3)  # Medium power
    inst.set_ramp(output=output, on=0, rate_value=1.0)  # No ramping
    
    return True

def set_setpoint_with_retry(inst, setpoint, output, max_attempts=5):
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

def wait_for_stability(inst, target_temp, tolerance=0.1, max_wait=30, check_interval=1):
    """Wait for temperature to stabilize within tolerance of target."""
    start_time = time.time()
    stable_count = 0
    required_stable_readings = 3
    
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

def perform_step_response_test(inst, config):
    """Perform step response test using configuration parameters."""
    
    # Get parameters from config
    start_temp = config['step_test']['start_temp']
    target_temp = config['step_test']['target_temp']
    output = config['instrument']['output_channel']
    recording_interval = config['step_test']['recording_interval']
    baseline_time = config['step_test']['baseline_time']
    step_hold_time = config['step_test']['step_hold_time']
    
    # Setup controller
    setup_controller(inst, output, config)
    
    # Set initial setpoint and wait for stability
    logging.info(f"Setting initial setpoint to {start_temp}K")
    set_setpoint_with_retry(inst, start_temp, output)
    
    logging.info(f"Waiting for stability at {start_temp}K...")
    stabilized, actual_temp, wait_time = wait_for_stability(
        inst, start_temp, tolerance=0.1, max_wait=300
    )
    
    if not stabilized:
        logging.warning(f"Did not stabilize at start temp. Current: {actual_temp:.3f}K")
    
    # Start data recording
    results = []
    test_start_time = time.time()
    step_applied = False
    
    logging.info(f"Starting step response test: {start_temp}K -> {target_temp}K")
    
    while True:
        current_time = time.time() - test_start_time
        
        # Apply step after baseline period
        if not step_applied and current_time > baseline_time:
            logging.info(f"Applying step to {target_temp}K")
            set_setpoint_with_retry(inst, target_temp, output)
            step_applied = True
            step_time = current_time
        
        # Stop test after step has had time to respond
        if step_applied and current_time > step_time + step_hold_time:
            logging.info("Step response test complete")
            break
        
        # Record data
        actual_temp = float(inst.read_temp(channel="A"))
        controller_setpoint = float(inst.get_setpoint(output=output))
        
        results.append({
            'time_s': current_time,
            'setpoint_K': controller_setpoint,
            'actual_temp_K': actual_temp,
            'error_mK': (actual_temp - controller_setpoint) * 1000,
            'step_applied': step_applied
        })
        
        # Periodic logging
        if len(results) % 20 == 0:
            logging.info(f"t={current_time:.0f}s: {actual_temp:.3f}K (setpoint: {controller_setpoint:.3f}K)")
        
        time.sleep(recording_interval)
    
    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze step response test results."""
    step_time = df[df['step_applied'] == True]['time_s'].iloc[0] if any(df['step_applied']) else None
    
    logging.info("STEP RESPONSE ANALYSIS")
    logging.info(f"Test duration: {df['time_s'].max():.1f}s")
    logging.info(f"Step applied at: {step_time:.1f}s" if step_time else "No step found")
    
    if step_time:
        initial_phase = df[df['time_s'] < step_time]
        response_phase = df[df['time_s'] >= step_time]
        
        logging.info(f"Initial stability: {initial_phase['actual_temp_K'].std():.3f}K std")
        logging.info(f"Target setpoint: {response_phase['setpoint_K'].iloc[0]:.3f}K")
        logging.info(f"Max temperature: {response_phase['actual_temp_K'].max():.3f}K")

def create_step_response_plot(df, filename_prefix, output_dir="step_response_results"):
    """Create simplified step response plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Temperature plot
    ax1.plot(df['time_s']/60, df['actual_temp_K'], 'b-', linewidth=2, label='Temperature')
    ax1.plot(df['time_s']/60, df['setpoint_K'], 'r--', linewidth=2, label='Setpoint')
    
    # Mark step
    if any(df['step_applied']):
        step_time = df[df['step_applied'] == True]['time_s'].iloc[0] / 60
        ax1.axvline(x=step_time, color='green', linestyle=':', label=f'Step ({step_time:.1f}min)')
    
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Step Response Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error plot
    ax2.plot(df['time_s']/60, df['error_mK'], 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Error (mK)')
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    logging.info(f"Plot saved: {plot_filename}")
    
    plt.show()
    return plot_filename

def main():
    """Main function to run step response test."""
    
    logging.info("Temperature Step Response Test - Lakeshore 336")
    
    # Load configuration
    config = load_config()
    if config is None:
        # Default configuration
        config = {
            'instrument': {'port': 'GPIB0::12::INSTR', 'output_channel': 1},
            'step_test': {
                'start_temp': 3.5,
                'target_temp': 6.0,
                'recording_interval': 0.5,
                'baseline_time': 30,
                'step_hold_time': 30
            }
        }
    
    # Initialize instrument
    logging.info("Connecting to Lakeshore 336...")
    inst = ctl.Lakeshore336(config['instrument']['port'])
    
    # Read initial temperature
    initial_temp = float(inst.read_temp(channel="A"))
    logging.info(f"Initial temperature: {initial_temp:.3f}K")
    
    # Show test configuration
    step_config = config['step_test']
    logging.info(f"Test: {step_config['start_temp']}K -> {step_config['target_temp']}K")
    
    # Perform the test
    results_df = perform_step_response_test(inst, config)
    
    # Analyze and save results
    analyze_results(results_df)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"step_response_{timestamp}.csv")
    results_df.to_csv(filename, index=False)
    logging.info(f"Data saved: {filename}")
    
    # Create plot
    create_step_response_plot(results_df, "step_response", output_dir)
    
    # Return to initial temperature
    logging.info(f"Returning to initial temperature ({initial_temp:.3f}K)")
    inst.set_setpoint(setpoint=initial_temp, output=config['instrument']['output_channel'])
    
    logging.info("Test complete!")
    return results_df

if __name__ == "__main__":
    results = main()

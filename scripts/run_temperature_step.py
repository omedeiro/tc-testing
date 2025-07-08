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
from temperature_control_utils import (
    load_config, setup_controller, set_setpoint_with_retry, wait_for_stability, shutdown_heater
)

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
    
    # Disable ramping for step response test
    inst.set_ramp(output=output, on=0, rate_value=1.0)  # No ramping
    
    # Set initial setpoint and wait for stability
    logging.info(f"Setting initial setpoint to {start_temp}K")
    set_setpoint_with_retry(inst, start_temp, output)
    
    logging.info(f"Waiting for stability at {start_temp}K...")
    stabilized, actual_temp, wait_time = wait_for_stability(
        inst, start_temp, tolerance=0.1, max_wait=300, check_interval=1
    )
    
    if not stabilized:
        logging.warning(f"Did not stabilize at start temp. Current: {actual_temp:.3f}K")
    
    # Start data recording
    results = []
    test_start_time = time.time()
    step_up_applied = False
    step_down_applied = False
    
    logging.info(f"Starting dual step response test: {start_temp}K -> {target_temp}K -> {start_temp}K")
    
    while True:
        current_time = time.time() - test_start_time
        
        # Apply up step after baseline period
        if not step_up_applied and current_time > baseline_time:
            logging.info(f"Applying up step to {target_temp}K")
            set_setpoint_with_retry(inst, target_temp, output)
            step_up_applied = True
            step_up_time = current_time
        
        # Apply down step after hold time at target
        if step_up_applied and not step_down_applied and current_time > step_up_time + step_hold_time:
            logging.info(f"Applying down step back to {start_temp}K")
            set_setpoint_with_retry(inst, start_temp, output)
            step_down_applied = True
            step_down_time = current_time
        
        # Stop test after down step has had time to respond
        if step_down_applied and current_time > step_down_time + step_hold_time:
            logging.info("Dual step response test complete")
            break
        
        # Record data
        actual_temp = float(inst.read_temp(channel="A"))
        controller_setpoint = float(inst.get_setpoint(output=output))
        
        # Determine test phase
        if not step_up_applied:
            phase = "baseline"
        elif not step_down_applied:
            phase = "step_up"
        else:
            phase = "step_down"
        
        results.append({
            'time_s': current_time,
            'setpoint_K': controller_setpoint,
            'actual_temp_K': actual_temp,
            'error_mK': (actual_temp - controller_setpoint) * 1000,
            'step_up_applied': step_up_applied,
            'step_down_applied': step_down_applied,
            'phase': phase
        })
        
        # Periodic logging
        if len(results) % 20 == 0:
            logging.info(f"t={current_time:.0f}s ({phase}): {actual_temp:.3f}K (setpoint: {controller_setpoint:.3f}K)")
        
        time.sleep(recording_interval)
    
    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze step response test results."""
    step_up_time = df[df['step_up_applied'] == True]['time_s'].iloc[0] if any(df['step_up_applied']) else None
    step_down_time = df[df['step_down_applied'] == True]['time_s'].iloc[0] if any(df['step_down_applied']) else None
    
    logging.info("DUAL STEP RESPONSE ANALYSIS")
    logging.info(f"Total test duration: {df['time_s'].max():.1f}s")
    logging.info(f"Up step applied at: {step_up_time:.1f}s" if step_up_time else "No up step found")
    logging.info(f"Down step applied at: {step_down_time:.1f}s" if step_down_time else "No down step found")
    
    # Analyze each phase
    phases = ['baseline', 'step_up', 'step_down']
    for phase in phases:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            duration = phase_data['time_s'].max() - phase_data['time_s'].min()
            temp_range = phase_data['actual_temp_K'].max() - phase_data['actual_temp_K'].min()
            avg_error = phase_data['error_mK'].mean()
            std_error = phase_data['error_mK'].std()
            
            logging.info(f"{phase.upper()}: {duration:.1f}s, ΔT={temp_range:.3f}K, "
                        f"error={avg_error:.1f}±{std_error:.1f}mK")
    
    # Analyze step responses
    if step_up_time:
        up_response = df[df['phase'] == 'step_up']
        if len(up_response) > 0:
            start_temp = up_response['actual_temp_K'].iloc[0]
            max_temp = up_response['actual_temp_K'].max()
            target_setpoint = up_response['setpoint_K'].iloc[-1]
            logging.info(f"Up step: {start_temp:.3f}K -> {max_temp:.3f}K (target: {target_setpoint:.3f}K)")
    
    if step_down_time:
        down_response = df[df['phase'] == 'step_down']
        if len(down_response) > 0:
            start_temp = down_response['actual_temp_K'].iloc[0]
            min_temp = down_response['actual_temp_K'].min()
            target_setpoint = down_response['setpoint_K'].iloc[-1]
            logging.info(f"Down step: {start_temp:.3f}K -> {min_temp:.3f}K (target: {target_setpoint:.3f}K)")

def create_step_response_plot(df, filename_prefix, config, output_dir="step_response_results"):
    """Create dual step response plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Color map for phases
    phase_colors = {
        'baseline': 'gray',
        'step_up': 'blue',
        'step_down': 'red'
    }
    
    # Temperature plot
    ax1.plot(df['time_s']/60, df['actual_temp_K'], 'k-', linewidth=2, label='Temperature')
    ax1.plot(df['time_s']/60, df['setpoint_K'], 'r--', linewidth=2, label='Setpoint')
    
    # Mark steps
    if any(df['step_up_applied']):
        step_up_time = df[df['step_up_applied'] == True]['time_s'].iloc[0] / 60
        ax1.axvline(x=step_up_time, color='green', linestyle=':', label=f'Up Step ({step_up_time:.1f}min)')
    
    if any(df['step_down_applied']):
        step_down_time = df[df['step_down_applied'] == True]['time_s'].iloc[0] / 60
        ax1.axvline(x=step_down_time, color='orange', linestyle=':', label=f'Down Step ({step_down_time:.1f}min)')
    
    # Color background by phase
    phases = df['phase'].unique()
    for phase in phases:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            ax1.axvspan(phase_data['time_s'].min()/60, phase_data['time_s'].max()/60, 
                       alpha=0.2, color=phase_colors.get(phase, 'gray'), 
                       label=f'{phase} phase')
    
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Dual Step Response Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error plot
    ax2.plot(df['time_s']/60, df['error_mK'], 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Mark steps in error plot
    if any(df['step_up_applied']):
        step_up_time = df[df['step_up_applied'] == True]['time_s'].iloc[0] / 60
        ax2.axvline(x=step_up_time, color='green', linestyle=':', alpha=0.7)
    
    if any(df['step_down_applied']):
        step_down_time = df[df['step_down_applied'] == True]['time_s'].iloc[0] / 60
        ax2.axvline(x=step_down_time, color='orange', linestyle=':', alpha=0.7)
    
    # Color background by phase
    for phase in phases:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            ax2.axvspan(phase_data['time_s'].min()/60, phase_data['time_s'].max()/60, 
                       alpha=0.2, color=phase_colors.get(phase, 'gray'))
    
    ax2.set_ylabel('Error (mK)')
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, alpha=0.3)
    
    # Add PID settings text box
    pid_config = config.get('pid', {}) if config else {}
    p_value = pid_config.get('P', 100)
    i_value = pid_config.get('I', 25)
    d_value = pid_config.get('D', 15)
    
    pid_text = f'PID Settings:\nP = {p_value}\nI = {i_value}\nD = {d_value}'
    ax1.text(0.02, 0.98, pid_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
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
    config = load_config("../configs/step_response_config.yaml")
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
    logging.info(f"Dual step test: {step_config['start_temp']}K -> {step_config['target_temp']}K -> {step_config['start_temp']}K")
    
    # Perform the test
    results_df = perform_step_response_test(inst, config)
    
    # Analyze and save results
    analyze_results(results_df)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"step_response_{timestamp}.csv")
    results_df.to_csv(filename, index=False)
    logging.info(f"Data saved: {filename}")
    
    # Create plot
    create_step_response_plot(results_df, "step_response", config, output_dir)
    
    # Shut down heater to allow return to base temperature
    shutdown_heater(inst, config['instrument']['output_channel'])
    
    logging.info("Test complete!")
    return results_df

if __name__ == "__main__":
    results = main()

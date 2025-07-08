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
    load_config, setup_controller, set_setpoint_with_retry, 
    set_setpoint_for_ramp, wait_for_stability, configure_ramp, shutdown_heater
)

# Create output directory
output_dir = "ramp_test_results"
os.makedirs(output_dir, exist_ok=True)

# Configure logging
log_filename = os.path.join(output_dir, 'temperature_ramp.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)

def perform_ramp_test(inst, config):
    """Perform temperature ramp test using configuration parameters."""
    
    # Get parameters from config
    start_temp = config['ramp_test']['start_temp']
    target_temp = config['ramp_test']['target_temp']
    ramp_rate = config['ramp_test']['ramp_rate']
    output = config['instrument']['output_channel']
    recording_interval = config['ramp_test']['recording_interval']
    hold_time = config['ramp_test']['hold_time']
    
    # Setup controller
    setup_controller(inst, output, config)
    
    # Set initial setpoint without ramping
    logging.info(f"Setting initial setpoint to {start_temp}K (no ramp)")
    configure_ramp(inst, output, ramp_rate, enable=False)
    set_setpoint_with_retry(inst, start_temp, output)
    
    logging.info(f"Waiting for stability at {start_temp}K...")
    stabilized, actual_temp, wait_time = wait_for_stability(
        inst, start_temp, tolerance=0.1, max_wait=300
    )
    
    if not stabilized:
        logging.warning(f"Did not stabilize at start temp. Current: {actual_temp:.3f}K")
    else:
        logging.info(f"Stabilized at {actual_temp:.3f}K after {wait_time:.1f}s")
    
    # Start data recording
    results = []
    test_start_time = time.time()
    ramp_up_started = False
    ramp_down_started = False
    hold_started = False
    
    logging.info(f"Starting ramp test: {start_temp}K -> {target_temp}K @ {ramp_rate:.2f} K/min")
    
    while True:
        current_time = time.time() - test_start_time
        
        # Start ramp up
        if not ramp_up_started and current_time > 10:  # Start ramp after 10s baseline
            logging.info(f"Starting ramp up to {target_temp}K")
            configure_ramp(inst, output, ramp_rate, enable=True)
            set_setpoint_for_ramp(inst, target_temp, output)
            ramp_up_started = True
            ramp_up_time = current_time
            logging.info(f"Ramp started - setpoint will gradually change to {target_temp}K")
        
        # Check if we've reached target and start hold phase
        if ramp_up_started and not hold_started:
            current_setpoint = float(inst.get_setpoint(output=output))
            if abs(current_setpoint - target_temp) < 0.05:  # Setpoint has reached target
                logging.info(f"Reached target setpoint ({current_setpoint:.3f}K), starting hold phase")
                hold_started = True
                hold_start_time = current_time
        
        # Start ramp down after hold time
        if hold_started and not ramp_down_started and current_time > hold_start_time + hold_time:
            logging.info(f"Starting ramp down to {start_temp}K")
            configure_ramp(inst, output, ramp_rate, enable=True)
            set_setpoint_for_ramp(inst, start_temp, output)
            ramp_down_started = True
            ramp_down_time = current_time
            logging.info(f"Ramp down started - setpoint will gradually change to {start_temp}K")
        
        # End test when ramp down is complete
        if ramp_down_started:
            current_setpoint = float(inst.get_setpoint(output=output))
            if abs(current_setpoint - start_temp) < 0.05:  # Setpoint has reached start temp
                logging.info(f"Ramp down complete - setpoint reached {current_setpoint:.3f}K")
                break
        
        # Record data
        actual_temp = float(inst.read_temp(channel="A"))
        controller_setpoint = float(inst.get_setpoint(output=output))
        
        # Determine test phase
        if not ramp_up_started:
            phase = "baseline"
        elif not hold_started:
            phase = "ramp_up"
        elif not ramp_down_started:
            phase = "hold"
        else:
            phase = "ramp_down"
        
        results.append({
            'time_s': current_time,
            'setpoint_K': controller_setpoint,
            'actual_temp_K': actual_temp,
            'error_mK': (actual_temp - controller_setpoint) * 1000,
            'phase': phase,
            'ramp_up_started': ramp_up_started,
            'hold_started': hold_started,
            'ramp_down_started': ramp_down_started
        })
        
        # Periodic logging
        if len(results) % 20 == 0:
            logging.info(f"t={current_time:.0f}s ({phase}): T={actual_temp:.3f}K, SP={controller_setpoint:.3f}K")
        
        time.sleep(recording_interval)
    
    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze ramp test results."""
    logging.info("RAMP TEST ANALYSIS")
    logging.info(f"Total test duration: {df['time_s'].max():.1f}s")
    
    # Analyze each phase
    phases = ['baseline', 'ramp_up', 'hold', 'ramp_down']
    for phase in phases:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            duration = phase_data['time_s'].max() - phase_data['time_s'].min()
            temp_range = phase_data['actual_temp_K'].max() - phase_data['actual_temp_K'].min()
            avg_error = phase_data['error_mK'].mean()
            std_error = phase_data['error_mK'].std()
            
            logging.info(f"{phase.upper()}: {duration:.1f}s, ΔT={temp_range:.3f}K, "
                        f"error={avg_error:.1f}±{std_error:.1f}mK")
    
    # Calculate ramp rates
    ramp_up_data = df[df['phase'] == 'ramp_up']
    if len(ramp_up_data) > 10:
        time_diff = ramp_up_data['time_s'].iloc[-1] - ramp_up_data['time_s'].iloc[0]
        temp_diff = ramp_up_data['actual_temp_K'].iloc[-1] - ramp_up_data['actual_temp_K'].iloc[0]
        actual_rate = (temp_diff / time_diff) * 60  # K/min
        logging.info(f"Actual ramp up rate: {actual_rate:.2f} K/min")
    
    ramp_down_data = df[df['phase'] == 'ramp_down']
    if len(ramp_down_data) > 10:
        time_diff = ramp_down_data['time_s'].iloc[-1] - ramp_down_data['time_s'].iloc[0]
        temp_diff = ramp_down_data['actual_temp_K'].iloc[-1] - ramp_down_data['actual_temp_K'].iloc[0]
        actual_rate = abs(temp_diff / time_diff) * 60  # K/min
        logging.info(f"Actual ramp down rate: {actual_rate:.2f} K/min")

def create_ramp_plot(df, filename_prefix, config, output_dir="ramp_test_results"):
    """Create comprehensive ramp test plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Color map for phases
    phase_colors = {
        'baseline': 'gray',
        'ramp_up': 'blue',
        'hold': 'green',
        'ramp_down': 'red'
    }
    
    # Temperature plot
    ax1.plot(df['time_s']/60, df['actual_temp_K'], 'k-', linewidth=2, label='Temperature')
    ax1.plot(df['time_s']/60, df['setpoint_K'], 'r--', linewidth=2, label='Setpoint')
    
    # Color background by phase
    phases = df['phase'].unique()
    for phase in phases:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            ax1.axvspan(phase_data['time_s'].min()/60, phase_data['time_s'].max()/60, 
                       alpha=0.2, color=phase_colors.get(phase, 'gray'), 
                       label=f'{phase} phase')
    
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Temperature Ramp Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error plot
    ax2.plot(df['time_s']/60, df['error_mK'], 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
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
    """Main function to run temperature ramp test."""
    
    logging.info("Temperature Ramp Test - Lakeshore 336")
    
    # Load configuration
    config = load_config("configs/ramp_test_config.yaml")
    if config is None:
        # Default configuration
        config = {
            'instrument': {'port': 'GPIB0::12::INSTR', 'output_channel': 1},
            'ramp_test': {
                'start_temp': 3.5,
                'target_temp': 6.0,
                'ramp_rate': 1.0,  # K/min
                'recording_interval': 1.0,  # seconds
                'hold_time': 60  # seconds at target temp
            }
        }
    
    # Initialize instrument
    logging.info("Connecting to Lakeshore 336...")
    inst = ctl.Lakeshore336(config['instrument']['port'])
    
    # Read initial temperature
    initial_temp = float(inst.read_temp(channel="A"))
    logging.info(f"Initial temperature: {initial_temp:.3f}K")
    
    # Show test configuration
    ramp_config = config['ramp_test']
    logging.info(f"Ramp test: {ramp_config['start_temp']}K -> {ramp_config['target_temp']}K @ {ramp_config['ramp_rate']:.2f} K/min")
    logging.info(f"Hold time: {ramp_config['hold_time']}s")
    
    # Perform the test
    results_df = perform_ramp_test(inst, config)
    
    # Analyze and save results
    analyze_results(results_df)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"ramp_test_{timestamp}.csv")
    results_df.to_csv(filename, index=False)
    logging.info(f"Data saved: {filename}")
    
    # Create plot
    create_ramp_plot(results_df, "ramp_test", config, output_dir)
    
    # Shut down heater to allow return to base temperature
    shutdown_heater(inst, config['instrument']['output_channel'])
    
    logging.info("Test complete!")
    return results_df

if __name__ == "__main__":
    results = main()

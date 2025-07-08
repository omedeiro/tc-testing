import qnnpy.functions.functions as qf
import qnnpy.instruments.lakeshore336 as ctl
from qnnpy.instruments.keithley_2400 import Keithley2400
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
output_dir = "resistance_ramp_results"
os.makedirs(output_dir, exist_ok=True)

# Configure logging
log_filename = os.path.join(output_dir, 'resistance_ramp.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)

def setup_sourcemeter(sourcemeter: Keithley2400, current_level=1e-6):
    """Setup Keithley 2400 for 2-wire resistance measurement."""
    logging.info("Setting up Keithley 2400 sourcemeter...")
    
    try:
        # Reset instrument
        sourcemeter.reset()
        time.sleep(0.5)
        
        # Configure for 2-wire current source, voltage measurement
        sourcemeter.setup_2W_source_I_read_V()
        
        # Set current level (1 µA)
        sourcemeter.set_current(current_level)
        logging.info(f"Current source level set to: {current_level*1e6:.1f} µA")
        
        # Set voltage compliance (10V default)
        sourcemeter.set_compliance_v(10.0)
        
        # Set measurement time for better accuracy
        sourcemeter.set_measurement_time(1.0)  # 1 PLC for good accuracy
        
        # Enable output but don't turn on yet
        logging.info("✓ Sourcemeter configured for 2-wire resistance measurement")
        return True
        
    except Exception as e:
        logging.error(f"✗ Failed to setup sourcemeter: {str(e)}")
        return False

def measure_resistance(sourcemeter, current_level=1e-6):
    """Measure voltage and calculate resistance."""
    try:
        # Measure voltage
        voltage = sourcemeter.read_voltage()
        
        # Calculate resistance using Ohm's law (V = I*R)
        resistance = voltage / current_level
        
        return voltage, resistance
        
    except Exception as e:
        logging.error(f"Error measuring resistance: {str(e)}")
        return None, None

def perform_resistance_ramp_test(temp_controller, sourcemeter, config):
    """Perform temperature ramp test with simultaneous resistance measurements."""
    
    # Get parameters from config
    start_temp = config['ramp_test']['start_temp']
    target_temp = config['ramp_test']['target_temp']
    ramp_rate = config['ramp_test']['ramp_rate']
    output = config['instrument']['output_channel']
    recording_interval = config['ramp_test']['recording_interval']
    hold_time = config['ramp_test']['hold_time']
    current_level = config['sourcemeter'].get('current_level', 1e-6)
    
    # Setup temperature controller
    setup_controller(temp_controller, output, config)
    
    # Setup sourcemeter
    if not setup_sourcemeter(sourcemeter, current_level):
        raise RuntimeError("Failed to setup sourcemeter")
    
    # Set initial setpoint without ramping
    logging.info(f"Setting initial setpoint to {start_temp}K (no ramp)")
    configure_ramp(temp_controller, output, ramp_rate, enable=False)
    set_setpoint_with_retry(temp_controller, start_temp, output)
    
    logging.info(f"Waiting for stability at {start_temp}K...")
    stabilized, actual_temp, wait_time = wait_for_stability(
        temp_controller, start_temp, tolerance=0.1, max_wait=300
    )
    
    if not stabilized:
        logging.warning(f"Did not stabilize at start temp. Current: {actual_temp:.3f}K")
    else:
        logging.info(f"Stabilized at {actual_temp:.3f}K after {wait_time:.1f}s")
    
    # Turn on sourcemeter output
    logging.info("Turning on sourcemeter output...")
    sourcemeter.set_output(True)
    time.sleep(2)  # Allow settling time
    
    # Start data recording
    results = []
    test_start_time = time.time()
    ramp_up_started = False
    ramp_down_started = False
    hold_started = False
    
    logging.info(f"Starting resistance ramp test: {start_temp}K -> {target_temp}K @ {ramp_rate:.2f} K/min")
    logging.info(f"Sourcemeter current: {current_level*1e6:.1f} µA")
    
    try:
        while True:
            current_time = time.time() - test_start_time
            
            # Start ramp up
            if not ramp_up_started and current_time > 10:  # Start ramp after 10s baseline
                logging.info(f"Starting ramp up to {target_temp}K")
                configure_ramp(temp_controller, output, ramp_rate, enable=True)
                set_setpoint_for_ramp(temp_controller, target_temp, output)
                ramp_up_started = True
                ramp_up_time = current_time
                logging.info(f"Ramp started - setpoint will gradually change to {target_temp}K")
            
            # Check if we've reached target and start hold phase
            if ramp_up_started and not hold_started:
                current_setpoint = float(temp_controller.get_setpoint(output=output))
                if abs(current_setpoint - target_temp) < 0.05:  # Setpoint has reached target
                    logging.info(f"Reached target setpoint ({current_setpoint:.3f}K), starting hold phase")
                    hold_started = True
                    hold_start_time = current_time
            
            # Start ramp down after hold time
            if hold_started and not ramp_down_started and current_time > hold_start_time + hold_time:
                logging.info(f"Starting ramp down to {start_temp}K")
                configure_ramp(temp_controller, output, ramp_rate, enable=True)
                set_setpoint_for_ramp(temp_controller, start_temp, output)
                ramp_down_started = True
                ramp_down_time = current_time
                logging.info(f"Ramp down started - setpoint will gradually change to {start_temp}K")
            
            # End test when ramp down is complete
            if ramp_down_started:
                current_setpoint = float(temp_controller.get_setpoint(output=output))
                if abs(current_setpoint - start_temp) < 0.05:  # Setpoint has reached start temp
                    logging.info(f"Ramp down complete - setpoint reached {current_setpoint:.3f}K")
                    break
            
            # Record temperature data
            actual_temp = float(temp_controller.read_temp(channel="A"))
            controller_setpoint = float(temp_controller.get_setpoint(output=output))
            
            # Record resistance data
            voltage, resistance = measure_resistance(sourcemeter, current_level)
            
            # Determine test phase
            if not ramp_up_started:
                phase = "baseline"
            elif not hold_started:
                phase = "ramp_up"
            elif not ramp_down_started:
                phase = "hold"
            else:
                phase = "ramp_down"
            
            # Store all data
            data_point = {
                'time_s': current_time,
                'setpoint_K': controller_setpoint,
                'actual_temp_K': actual_temp,
                'temp_error_mK': (actual_temp - controller_setpoint) * 1000,
                'voltage_V': voltage if voltage is not None else np.nan,
                'resistance_ohm': resistance if resistance is not None else np.nan,
                'current_A': current_level,
                'phase': phase,
                'ramp_up_started': ramp_up_started,
                'hold_started': hold_started,
                'ramp_down_started': ramp_down_started
            }
            
            results.append(data_point)
            
            # Periodic logging
            if len(results) % 20 == 0:
                if resistance is not None:
                    logging.info(f"t={current_time:.0f}s ({phase}): T={actual_temp:.3f}K, SP={controller_setpoint:.3f}K, "
                               f"V={voltage:.6f}V, R={resistance:.3f}Ω")
                else:
                    logging.info(f"t={current_time:.0f}s ({phase}): T={actual_temp:.3f}K, SP={controller_setpoint:.3f}K, "
                               f"R=ERROR")
            
            time.sleep(recording_interval)
    
    finally:
        # Turn off sourcemeter output
        logging.info("Turning off sourcemeter output...")
        sourcemeter.set_output(False)
    
    return pd.DataFrame(results)

def analyze_resistance_results(df):
    """Analyze resistance ramp test results."""
    logging.info("RESISTANCE RAMP TEST ANALYSIS")
    logging.info(f"Total test duration: {df['time_s'].max():.1f}s")
    
    # Analyze each phase
    phases = ['baseline', 'ramp_up', 'hold', 'ramp_down']
    for phase in phases:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            duration = phase_data['time_s'].max() - phase_data['time_s'].min()
            temp_range = phase_data['actual_temp_K'].max() - phase_data['actual_temp_K'].min()
            avg_temp_error = phase_data['temp_error_mK'].mean()
            std_temp_error = phase_data['temp_error_mK'].std()
            
            # Resistance statistics (excluding NaN values)
            valid_resistance = phase_data['resistance_ohm'].dropna()
            if len(valid_resistance) > 0:
                avg_resistance = valid_resistance.mean()
                std_resistance = valid_resistance.std()
                resistance_range = valid_resistance.max() - valid_resistance.min()
                
                logging.info(f"{phase.upper()}: {duration:.1f}s, ΔT={temp_range:.3f}K, "
                           f"temp_error={avg_temp_error:.1f}±{std_temp_error:.1f}mK, "
                           f"R={avg_resistance:.3f}±{std_resistance:.3f}Ω (ΔR={resistance_range:.3f}Ω)")
            else:
                logging.info(f"{phase.upper()}: {duration:.1f}s, ΔT={temp_range:.3f}K, "
                           f"temp_error={avg_temp_error:.1f}±{std_temp_error:.1f}mK, R=NO_DATA")
    
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
    
    # Temperature coefficient analysis
    valid_data = df.dropna(subset=['resistance_ohm'])
    if len(valid_data) > 10:
        # Calculate temperature coefficient (dR/dT) for normal state
        temp_coeff = np.polyfit(valid_data['actual_temp_K'], valid_data['resistance_ohm'], 1)[0]
        logging.info(f"Overall temperature coefficient: {temp_coeff:.6f} Ω/K")
    
    # Critical temperature analysis
    logging.info("=" * 50)
    logging.info("CRITICAL TEMPERATURE ANALYSIS - FULL DATA")
    best_tc_full, tc_methods_full, tc_stats_full = analyze_critical_temperature(df)
    
    logging.info("=" * 50)
    logging.info("CRITICAL TEMPERATURE ANALYSIS - RAMP UP ONLY")
    ramp_up_data = df[df['phase'] == 'ramp_up']
    if len(ramp_up_data) > 10:
        best_tc_up, tc_methods_up, tc_stats_up = analyze_critical_temperature(ramp_up_data)
    else:
        logging.warning("Insufficient ramp up data for Tc analysis")
        best_tc_up, tc_methods_up, tc_stats_up = None, None, None
    
    logging.info("=" * 50)
    logging.info("CRITICAL TEMPERATURE ANALYSIS - RAMP DOWN ONLY")
    ramp_down_data = df[df['phase'] == 'ramp_down']
    if len(ramp_down_data) > 10:
        best_tc_down, tc_methods_down, tc_stats_down = analyze_critical_temperature(ramp_down_data)
    else:
        logging.warning("Insufficient ramp down data for Tc analysis")
        best_tc_down, tc_methods_down, tc_stats_down = None, None, None
    
    # Summary of hysteresis analysis
    logging.info("=" * 50)
    logging.info("HYSTERESIS ANALYSIS SUMMARY")
    if best_tc_up is not None and best_tc_down is not None:
        tc_hysteresis = best_tc_up - best_tc_down
        logging.info(f"Tc (ramp up): {best_tc_up:.3f} K ({tc_stats_up['best_method']})")
        logging.info(f"Tc (ramp down): {best_tc_down:.3f} K ({tc_stats_down['best_method']})")
        logging.info(f"Hysteresis (Tc_up - Tc_down): {tc_hysteresis:.3f} K")
        
        if abs(tc_hysteresis) > 0.01:  # Significant hysteresis threshold
            logging.info(f"Significant hysteresis detected: {tc_hysteresis:.3f} K")
        else:
            logging.info("No significant hysteresis detected")
    else:
        logging.warning("Cannot calculate hysteresis - insufficient data for one or both ramps")
    logging.info("=" * 50)

def create_resistance_ramp_plot(df, filename_prefix, config, output_dir="resistance_ramp_results"):
    """Create comprehensive resistance ramp test plot."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
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
    ax1.set_title('Temperature and Resistance Ramp Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Resistance plot
    valid_resistance = df.dropna(subset=['resistance_ohm'])
    if len(valid_resistance) > 0:
        ax2.plot(valid_resistance['time_s']/60, valid_resistance['resistance_ohm'], 'b-', linewidth=2, label='Resistance')
    
    # Color background by phase
    for phase in phases:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            ax2.axvspan(phase_data['time_s'].min()/60, phase_data['time_s'].max()/60, 
                       alpha=0.2, color=phase_colors.get(phase, 'gray'))
    
    ax2.set_ylabel('Resistance (Ω)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Temperature error plot
    ax3.plot(df['time_s']/60, df['temp_error_mK'], 'g-', linewidth=1.5, label='Temperature Error')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Color background by phase
    for phase in phases:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            ax3.axvspan(phase_data['time_s'].min()/60, phase_data['time_s'].max()/60, 
                       alpha=0.2, color=phase_colors.get(phase, 'gray'))
    
    ax3.set_ylabel('Temperature Error (mK)')
    ax3.set_xlabel('Time (minutes)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add configuration info text box
    pid_config = config.get('pid', {}) if config else {}
    p_value = pid_config.get('P', 100)
    i_value = pid_config.get('I', 25)
    d_value = pid_config.get('D', 15)
    
    sourcemeter_config = config.get('sourcemeter', {}) if config else {}
    current_level = sourcemeter_config.get('current_level', 1e-6)
    
    info_text = f'PID Settings:\nP = {p_value}\nI = {i_value}\nD = {d_value}\n\nSourcemeter:\nI = {current_level*1e6:.1f} µA'
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    logging.info(f"Plot saved: {plot_filename}")
    
    plt.show()
    return plot_filename

def create_resistance_vs_temperature_plot(df, filename_prefix, config, output_dir="resistance_ramp_results"):
    """Create resistance vs temperature plot with critical temperature analysis."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot resistance vs temperature
    valid_data = df.dropna(subset=['resistance_ohm'])
    if len(valid_data) > 0:
        # Color by phase
        phase_colors = {'baseline': 'gray', 'ramp_up': 'blue', 'hold': 'green', 'ramp_down': 'red'}
        
        for phase in valid_data['phase'].unique():
            phase_data = valid_data[valid_data['phase'] == phase]
            ax.scatter(phase_data['actual_temp_K'], phase_data['resistance_ohm'], 
                      c=phase_colors.get(phase, 'black'), label=f'{phase} phase', alpha=0.7, s=20)
        
        # Perform critical temperature analysis
        best_tc, tc_methods, tc_stats = analyze_critical_temperature(df)
        
        # Analyze ramp up and ramp down separately for hysteresis
        ramp_up_data = df[df['phase'] == 'ramp_up']
        ramp_down_data = df[df['phase'] == 'ramp_down']
        
        best_tc_up = None
        best_tc_down = None
        
        if len(ramp_up_data) > 10:
            best_tc_up, _, _ = analyze_critical_temperature(ramp_up_data)
        
        if len(ramp_down_data) > 10:
            best_tc_down, _, _ = analyze_critical_temperature(ramp_down_data)
        
        # Mark critical temperatures on plot
        if best_tc is not None:
            ax.axvline(x=best_tc, color='red', linestyle='--', linewidth=2, 
                      label=f'Tc (overall) = {best_tc:.3f} K ({tc_stats["best_method"]})')
        
        if best_tc_up is not None:
            ax.axvline(x=best_tc_up, color='blue', linestyle='-.', linewidth=2, alpha=0.8,
                      label=f'Tc (ramp up) = {best_tc_up:.3f} K')
        
        if best_tc_down is not None:
            ax.axvline(x=best_tc_down, color='red', linestyle=':', linewidth=2, alpha=0.8,
                      label=f'Tc (ramp down) = {best_tc_down:.3f} K')
        
        # Mark other Tc estimates from overall analysis if available
        if best_tc is not None:
            colors = ['orange', 'purple', 'brown', 'pink']
            method_names = {'derivative': 'dR/dT max', 'threshold_10pct': '10% threshold', 
                           'midpoint_50pct': '50% midpoint', 'linear_onset': 'linear onset'}
            
            color_idx = 0
            for method, tc_value in tc_methods.items():
                if tc_value is not None and tc_value != best_tc:
                    ax.axvline(x=tc_value, color=colors[color_idx % len(colors)], 
                              linestyle=':', alpha=0.5, linewidth=1,
                              label=f'{method_names.get(method, method)}: {tc_value:.3f} K')
                    color_idx += 1
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Resistance (Ω)')
        ax.set_title('Resistance vs Temperature - Critical Temperature Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add critical temperature info
        if best_tc is not None and tc_stats is not None:
            info_text = f'Critical Temperature Analysis:\n'
            info_text += f'Overall Tc: {best_tc:.3f} K ({tc_stats["best_method"]})\n'
            
            # Add hysteresis information
            if best_tc_up is not None and best_tc_down is not None:
                tc_hysteresis = best_tc_up - best_tc_down
                info_text += f'Tc (ramp up): {best_tc_up:.3f} K\n'
                info_text += f'Tc (ramp down): {best_tc_down:.3f} K\n'
                info_text += f'Hysteresis: {tc_hysteresis:.3f} K\n'
            elif best_tc_up is not None:
                info_text += f'Tc (ramp up): {best_tc_up:.3f} K\n'
            elif best_tc_down is not None:
                info_text += f'Tc (ramp down): {best_tc_down:.3f} K\n'
            
            if tc_stats["std_tc"] > 0:
                info_text += f'Mean ± Std: {tc_stats["mean_tc"]:.3f} ± {tc_stats["std_tc"]:.3f} K\n'
            info_text += f'R range: {tc_stats["min_resistance"]:.2f} - {tc_stats["max_resistance"]:.2f} Ω\n'
            if tc_stats.get("temp_transition_width") is not None:
                info_text += f'Transition width: {tc_stats["temp_transition_width"]:.3f} K'
            else:
                info_text += f'Transition width: N/A'
            
            ax.text(0.02, 0.98, info_text, 
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            # Fallback to simple linear fit if Tc analysis fails
            coeffs = np.polyfit(valid_data['actual_temp_K'], valid_data['resistance_ohm'], 1)
            temp_range = np.linspace(valid_data['actual_temp_K'].min(), valid_data['actual_temp_K'].max(), 100)
            ax.plot(temp_range, np.polyval(coeffs, temp_range), 'k--', linewidth=2, 
                    label=f'Linear fit: R = {coeffs[0]:.6f}·T + {coeffs[1]:.6f}')
            
            temp_coeff = coeffs[0]
            ax.text(0.02, 0.98, f'Temperature Coefficient:\n{temp_coeff:.6f} Ω/K', 
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"{filename_prefix}_vs_temp_{timestamp}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    logging.info(f"R vs T plot saved: {plot_filename}")
    
    plt.show()
    return plot_filename

def analyze_critical_temperature(df):
    """Analyze resistance data to identify critical temperature (Tc)."""
    logging.info("CRITICAL TEMPERATURE ANALYSIS")
    
    # Get valid resistance data
    valid_data = df.dropna(subset=['resistance_ohm', 'actual_temp_K']).copy()
    if len(valid_data) < 10:
        logging.warning("Insufficient data for critical temperature analysis")
        return None, None, None
    
    # Sort by temperature for proper analysis
    valid_data = valid_data.sort_values('actual_temp_K')
    temperatures = valid_data['actual_temp_K'].values
    resistances = valid_data['resistance_ohm'].values
    
    # Method 1: Derivative method - find maximum dR/dT
    if len(temperatures) > 5:
        # Calculate derivative (dR/dT) using central differences
        dR_dT = np.gradient(resistances, temperatures)
        
        # Find temperature of maximum derivative
        max_derivative_idx = np.argmax(dR_dT)
        tc_derivative = temperatures[max_derivative_idx]
        max_derivative_value = dR_dT[max_derivative_idx]
        
        logging.info(f"Method 1 (Max dR/dT): Tc = {tc_derivative:.3f} K, dR/dT = {max_derivative_value:.2e} Ω/K")
    else:
        tc_derivative = None
        max_derivative_value = None
        logging.warning("Insufficient data points for derivative method")
    
    # Method 2: Threshold method - find where resistance crosses a threshold
    # Define superconducting threshold (e.g., 10% of maximum resistance)
    max_resistance = np.max(resistances)
    min_resistance = np.min(resistances)
    resistance_range = max_resistance - min_resistance
    
    # Use 10% of the resistance range above minimum as threshold
    threshold_resistance = min_resistance + 0.1 * resistance_range
    
    # Find first temperature where resistance exceeds threshold
    above_threshold = resistances > threshold_resistance
    if np.any(above_threshold):
        threshold_idx = np.where(above_threshold)[0][0]
        tc_threshold = temperatures[threshold_idx]
        logging.info(f"Method 2 (10% threshold): Tc = {tc_threshold:.3f} K, R_threshold = {threshold_resistance:.3f} Ω")
    else:
        tc_threshold = None
        logging.warning("No data points above resistance threshold")
    
    # Method 3: Midpoint method - find temperature at 50% of resistance transition
    midpoint_resistance = min_resistance + 0.5 * resistance_range
    
    # Find temperature closest to midpoint resistance
    midpoint_diff = np.abs(resistances - midpoint_resistance)
    midpoint_idx = np.argmin(midpoint_diff)
    tc_midpoint = temperatures[midpoint_idx]
    
    logging.info(f"Method 3 (50% midpoint): Tc = {tc_midpoint:.3f} K, R_midpoint = {midpoint_resistance:.3f} Ω")
    
    # Method 4: Onset method - extrapolate linear regions
    try:
        # Find superconducting region (low resistance, relatively flat)
        # Assume first 30% of temperature range is superconducting
        n_points = len(temperatures)
        sc_end_idx = max(3, int(0.3 * n_points))
        
        # Find normal region (high resistance, after transition)
        # Assume last 30% of temperature range is normal
        normal_start_idx = min(n_points - 3, int(0.7 * n_points))
        
        if sc_end_idx < normal_start_idx:
            # Fit lines to superconducting and normal regions
            sc_temps = temperatures[:sc_end_idx]
            sc_resistances = resistances[:sc_end_idx]
            normal_temps = temperatures[normal_start_idx:]
            normal_resistances = resistances[normal_start_idx:]
            
            if len(sc_temps) >= 2 and len(normal_temps) >= 2:
                # Linear fits
                sc_fit = np.polyfit(sc_temps, sc_resistances, 1)
                normal_fit = np.polyfit(normal_temps, normal_resistances, 1)
                
                # Find intersection (onset Tc)
                # sc_fit[0] * T + sc_fit[1] = normal_fit[0] * T + normal_fit[1]
                if abs(sc_fit[0] - normal_fit[0]) > 1e-10:  # Avoid division by zero
                    tc_onset = (normal_fit[1] - sc_fit[1]) / (sc_fit[0] - normal_fit[0])
                    
                    # Check if intersection is within reasonable range
                    if temperatures[0] <= tc_onset <= temperatures[-1]:
                        logging.info(f"Method 4 (Linear onset): Tc = {tc_onset:.3f} K")
                    else:
                        tc_onset = None
                        logging.warning("Linear onset method: intersection outside temperature range")
                else:
                    tc_onset = None
                    logging.warning("Linear onset method: parallel lines, no intersection")
            else:
                tc_onset = None
                logging.warning("Linear onset method: insufficient data points")
        else:
            tc_onset = None
            logging.warning("Linear onset method: overlapping regions")
    except Exception as e:
        tc_onset = None
        logging.warning(f"Linear onset method failed: {str(e)}")
    
    # Summary of results
    tc_methods = {
        'derivative': tc_derivative,
        'threshold_10pct': tc_threshold,
        'midpoint_50pct': tc_midpoint,
        'linear_onset': tc_onset
    }
    
    # Calculate temperature transition width (10% to 90% of resistance transition)
    temp_transition_width = None
    if tc_threshold is not None:
        # Find 90% threshold temperature
        threshold_90pct = min_resistance + 0.9 * resistance_range
        above_90pct = resistances > threshold_90pct
        if np.any(above_90pct):
            threshold_90_idx = np.where(above_90pct)[0][0]
            tc_90pct = temperatures[threshold_90_idx]
            temp_transition_width = tc_90pct - tc_threshold
            logging.info(f"Temperature transition width (10%-90%): {temp_transition_width:.3f} K")
        else:
            logging.warning("Could not find 90% threshold for transition width calculation")
    
    # Calculate statistics
    valid_tc_values = [tc for tc in tc_methods.values() if tc is not None]
    if valid_tc_values:
        mean_tc = np.mean(valid_tc_values)
        std_tc = np.std(valid_tc_values) if len(valid_tc_values) > 1 else 0
        
        logging.info("CRITICAL TEMPERATURE SUMMARY:")
        logging.info(f"Available methods: {len(valid_tc_values)}")
        logging.info(f"Mean Tc: {mean_tc:.3f} ± {std_tc:.3f} K")
        logging.info(f"Resistance range: {min_resistance:.3f} - {max_resistance:.3f} Ω")
        if temp_transition_width is not None:
            logging.info(f"Temperature transition width (10%-90%): {temp_transition_width:.3f} K")
        
        # Choose best estimate (prefer 50% midpoint method due to better noise immunity)
        if tc_midpoint is not None:
            best_tc = tc_midpoint
            best_method = "midpoint_50pct"
        elif tc_onset is not None:
            best_tc = tc_onset
            best_method = "linear_onset"
        elif tc_threshold is not None:
            best_tc = tc_threshold
            best_method = "threshold_10pct"
        elif tc_derivative is not None:
            best_tc = tc_derivative
            best_method = "derivative"
        else:
            best_tc = None
            best_method = None
        
        logging.info(f"Best estimate: Tc = {best_tc:.3f} K (method: {best_method})")
        
        return best_tc, tc_methods, {
            'min_resistance': min_resistance,
            'max_resistance': max_resistance,
            'resistance_range': resistance_range,
            'temp_transition_width': temp_transition_width,
            'mean_tc': mean_tc,
            'std_tc': std_tc,
            'best_method': best_method
        }
    else:
        logging.error("No valid critical temperature estimates found")
        return None, tc_methods, None

def create_critical_temperature_analysis_plot(df, filename_prefix, config, output_dir="resistance_ramp_results"):
    """Create detailed critical temperature analysis plot."""
    valid_data = df.dropna(subset=['resistance_ohm', 'actual_temp_K']).copy()
    if len(valid_data) < 10:
        logging.warning("Insufficient data for critical temperature analysis plot")
        return None
    
    # Sort by temperature
    valid_data = valid_data.sort_values('actual_temp_K')
    temperatures = valid_data['actual_temp_K'].values
    resistances = valid_data['resistance_ohm'].values
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: R vs T with all Tc estimates
    phase_colors = {'baseline': 'gray', 'ramp_up': 'blue', 'hold': 'green', 'ramp_down': 'red'}
    for phase in valid_data['phase'].unique():
        phase_data = valid_data[valid_data['phase'] == phase]
        ax1.scatter(phase_data['actual_temp_K'], phase_data['resistance_ohm'], 
                   c=phase_colors.get(phase, 'black'), label=f'{phase} phase', alpha=0.7, s=15)
    
    # Get Tc analysis
    best_tc, tc_methods, tc_stats = analyze_critical_temperature(df)
    
    if best_tc is not None:
        # Mark all Tc estimates
        colors = ['red', 'orange', 'purple', 'brown']
        linestyles = ['--', ':', '-.', '-']
        method_names = {'derivative': 'dR/dT max', 'threshold_10pct': '10% threshold', 
                       'midpoint_50pct': '50% midpoint', 'linear_onset': 'linear onset'}
        
        for i, (method, tc_value) in enumerate(tc_methods.items()):
            if tc_value is not None:
                style = '--' if method == tc_stats["best_method"] else ':'
                width = 2 if method == tc_stats["best_method"] else 1
                ax1.axvline(x=tc_value, color=colors[i % len(colors)], linestyle=style, 
                           linewidth=width, label=f'{method_names.get(method, method)}: {tc_value:.3f} K')
    
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title('Resistance vs Temperature')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale R vs T
    ax2.semilogy(temperatures, resistances, 'b-', linewidth=2, label='Resistance')
    if best_tc is not None:
        ax2.axvline(x=best_tc, color='red', linestyle='--', linewidth=2, 
                   label=f'Best Tc = {best_tc:.3f} K')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Resistance (Ω) [log scale]')
    ax2.set_title('Resistance vs Temperature (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: dR/dT vs T
    if len(temperatures) > 5:
        dR_dT = np.gradient(resistances, temperatures)
        ax3.plot(temperatures, dR_dT, 'g-', linewidth=2, label='dR/dT')
        
        # Mark maximum derivative
        max_idx = np.argmax(dR_dT)
        ax3.plot(temperatures[max_idx], dR_dT[max_idx], 'ro', markersize=8, 
                label=f'Max dR/dT at {temperatures[max_idx]:.3f} K')
        ax3.axvline(x=temperatures[max_idx], color='red', linestyle=':', alpha=0.7)
        
        ax3.set_xlabel('Temperature (K)')
        ax3.set_ylabel('dR/dT (Ω/K)')
        ax3.set_title('Temperature Derivative of Resistance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Normalized resistance showing transition
    if tc_stats is not None:
        min_r = tc_stats['min_resistance']
        max_r = tc_stats['max_resistance']
        normalized_r = (resistances - min_r) / (max_r - min_r)
        
        ax4.plot(temperatures, normalized_r, 'b-', linewidth=2, label='Normalized R')
        ax4.axhline(y=0.1, color='orange', linestyle=':', label='10% threshold')
        ax4.axhline(y=0.5, color='purple', linestyle=':', label='50% midpoint')
        ax4.axhline(y=0.9, color='brown', linestyle=':', label='90% threshold')
        
        # Mark Tc estimates
        for method, tc_value in tc_methods.items():
            if tc_value is not None:
                ax4.axvline(x=tc_value, color='red', linestyle=':', alpha=0.5)
        
        ax4.set_xlabel('Temperature (K)')
        ax4.set_ylabel('Normalized Resistance')
        ax4.set_title('Normalized Resistance Transition')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"{filename_prefix}_tc_analysis_{timestamp}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    logging.info(f"Tc analysis plot saved: {plot_filename}")
    
    plt.show()
    return plot_filename

def create_hysteresis_analysis_plot(df, filename_prefix, config, output_dir="resistance_ramp_results"):
    """Create a plot showing hysteresis in the superconducting transition."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get separate phase data
    ramp_up_data = df[df['phase'] == 'ramp_up'].dropna(subset=['resistance_ohm', 'actual_temp_K'])
    ramp_down_data = df[df['phase'] == 'ramp_down'].dropna(subset=['resistance_ohm', 'actual_temp_K'])
    
    if len(ramp_up_data) < 5 or len(ramp_down_data) < 5:
        logging.warning("Insufficient data for hysteresis analysis plot")
        return None
    
    # Plot 1: Overlaid R vs T for ramp up and ramp down
    ax1.scatter(ramp_up_data['actual_temp_K'], ramp_up_data['resistance_ohm'], 
               c='blue', alpha=0.7, s=20, label='Ramp up')
    ax1.scatter(ramp_down_data['actual_temp_K'], ramp_down_data['resistance_ohm'], 
               c='red', alpha=0.7, s=20, label='Ramp down')
    
    # Analyze Tc for each phase
    best_tc_up = None
    best_tc_down = None
    tc_hysteresis = None
    
    if len(ramp_up_data) > 10:
        best_tc_up, _, tc_stats_up = analyze_critical_temperature(ramp_up_data)
        if best_tc_up is not None:
            ax1.axvline(x=best_tc_up, color='blue', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Tc (up) = {best_tc_up:.3f} K')
    
    if len(ramp_down_data) > 10:
        best_tc_down, _, tc_stats_down = analyze_critical_temperature(ramp_down_data)
        if best_tc_down is not None:
            ax1.axvline(x=best_tc_down, color='red', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Tc (down) = {best_tc_down:.3f} K')
    
    # Calculate hysteresis
    if best_tc_up is not None and best_tc_down is not None:
        tc_hysteresis = best_tc_up - best_tc_down
        
        # Shade the hysteresis region
        if tc_hysteresis != 0:
            ax1.axvspan(min(best_tc_up, best_tc_down), max(best_tc_up, best_tc_down), 
                       alpha=0.2, color='yellow', label=f'Hysteresis: {tc_hysteresis:.3f} K')
    
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title('Hysteresis in Superconducting Transition')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Direct comparison of normalized transitions
    if len(ramp_up_data) > 10 and len(ramp_down_data) > 10:
        # Sort data by temperature
        ramp_up_sorted = ramp_up_data.sort_values('actual_temp_K')
        ramp_down_sorted = ramp_down_data.sort_values('actual_temp_K')
        
        # Normalize resistance for each phase
        up_min = ramp_up_sorted['resistance_ohm'].min()
        up_max = ramp_up_sorted['resistance_ohm'].max()
        up_normalized = (ramp_up_sorted['resistance_ohm'] - up_min) / (up_max - up_min)
        
        down_min = ramp_down_sorted['resistance_ohm'].min()
        down_max = ramp_down_sorted['resistance_ohm'].max()
        down_normalized = (ramp_down_sorted['resistance_ohm'] - down_min) / (down_max - down_min)
        
        ax2.plot(ramp_up_sorted['actual_temp_K'], up_normalized, 'b-', linewidth=2, 
                label='Ramp up (normalized)', alpha=0.8)
        ax2.plot(ramp_down_sorted['actual_temp_K'], down_normalized, 'r-', linewidth=2, 
                label='Ramp down (normalized)', alpha=0.8)
        
        # Mark critical temperatures
        if best_tc_up is not None:
            ax2.axvline(x=best_tc_up, color='blue', linestyle='--', alpha=0.8)
        if best_tc_down is not None:
            ax2.axvline(x=best_tc_down, color='red', linestyle='--', alpha=0.8)
        
        # Add threshold lines
        ax2.axhline(y=0.1, color='gray', linestyle=':', alpha=0.5, label='10% threshold')
        ax2.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='50% midpoint')
        ax2.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
        
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Normalized Resistance')
        ax2.set_title('Normalized Transition Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)
    
    # Add summary text
    summary_text = 'Hysteresis Analysis:\n'
    if best_tc_up is not None:
        summary_text += f'Tc (ramp up): {best_tc_up:.3f} K\n'
    if best_tc_down is not None:
        summary_text += f'Tc (ramp down): {best_tc_down:.3f} K\n'
    if tc_hysteresis is not None:
        summary_text += f'Hysteresis: {tc_hysteresis:.3f} K\n'
        if abs(tc_hysteresis) > 0.01:
            summary_text += 'Significant hysteresis detected'
        else:
            summary_text += 'No significant hysteresis'
    else:
        summary_text += 'Cannot calculate hysteresis'
    
    fig.suptitle('Superconducting Transition Hysteresis Analysis', fontsize=14, y=0.98)
    
    # Add text box
    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"{filename_prefix}_hysteresis_{timestamp}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    logging.info(f"Hysteresis analysis plot saved: {plot_filename}")
    
    plt.show()
    return plot_filename

def main():
    """Main function to run resistance ramp test."""
    
    logging.info("Resistance Ramp Test - Lakeshore 336 + Keithley 2400")
    
    # Load configuration
    config = load_config("../configs/resistance_ramp_config.yaml")
    if config is None:
        # Default configuration
        config = {
            'instrument': {'port': 'GPIB0::12::INSTR', 'output_channel': 1},
            'sourcemeter': {'port': 'GPIB0::23::INSTR', 'current_level': 1e-6},  # 1 µA
            'ramp_test': {
                'start_temp': 3.5,
                'target_temp': 6.0,
                'ramp_rate': 1.0,  # K/min
                'recording_interval': 1.0,  # seconds
                'hold_time': 60  # seconds at target temp
            },
            'pid': {
                'P': 100,
                'I': 25,
                'D': 15
            }
        }
    
    # Initialize instruments with timeout handling
    logging.info("Connecting to instruments...")
    try:
        # Temperature controller
        temp_controller = ctl.Lakeshore336(config['instrument']['port'])
        temp_controller.pyvisa.timeout = 10000  # 10 seconds
        
        # Sourcemeter
        sourcemeter = Keithley2400(config['sourcemeter']['port'])
        sourcemeter.pyvisa.timeout = 10000  # 10 seconds
        
        # Test connections
        logging.info("Testing instrument connections...")
        initial_temp = float(temp_controller.read_temp(channel="A"))
        logging.info(f"✓ Temperature controller connected! Initial temperature: {initial_temp:.3f}K")
        
        # Test sourcemeter
        sourcemeter_id = sourcemeter.query("*IDN?")
        logging.info(f"✓ Sourcemeter connected: {sourcemeter_id.strip()}")
        
        current_setpoint = float(temp_controller.get_setpoint(output=config['instrument']['output_channel']))
        logging.info(f"✓ Current setpoint: {current_setpoint:.3f}K")
        
    except Exception as e:

        logging.error(f"✗ Instrument connection failed: {str(e)}")
        logging.info("Troubleshooting tips:")
        logging.info("1. Check GPIB cable connections")
        logging.info("2. Verify instrument addresses in config file")
        logging.info("3. Make sure no other software is using the instruments")
        logging.info("4. Try power cycling the instruments")
        return None
    
    # Show test configuration
    ramp_config = config['ramp_test']
    sourcemeter_config = config['sourcemeter']
    logging.info(f"Ramp test: {ramp_config['start_temp']}K -> {ramp_config['target_temp']}K @ {ramp_config['ramp_rate']:.2f} K/min")
    logging.info(f"Hold time: {ramp_config['hold_time']}s")
    logging.info(f"Sourcemeter current: {sourcemeter_config['current_level']*1e6:.1f} µA")
    
    try:
        # Perform the test
        results_df = perform_resistance_ramp_test(temp_controller, sourcemeter, config)
        
        # Analyze and save results
        analyze_resistance_results(results_df)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"resistance_ramp_{timestamp}.csv")
        results_df.to_csv(filename, index=False)
        logging.info(f"Data saved: {filename}")
        
        # Create plots
        create_resistance_ramp_plot(results_df, "resistance_ramp", config, output_dir)
        create_resistance_vs_temperature_plot(results_df, "resistance", config, output_dir)
        create_critical_temperature_analysis_plot(results_df, "resistance", config, output_dir)
        create_hysteresis_analysis_plot(results_df, "resistance", config, output_dir)
        create_hysteresis_analysis_plot(results_df, "resistance", config, output_dir)
        
        # Analyze critical temperature with hysteresis
        tc_result, tc_methods, tc_stats = analyze_critical_temperature(results_df)
        
        # Also analyze individual ramp phases
        ramp_up_data = results_df[results_df['phase'] == 'ramp_up']
        ramp_down_data = results_df[results_df['phase'] == 'ramp_down']
        
        tc_up = None
        tc_down = None
        
        if len(ramp_up_data) > 10:
            tc_up, _, _ = analyze_critical_temperature(ramp_up_data)
        
        if len(ramp_down_data) > 10:
            tc_down, _, _ = analyze_critical_temperature(ramp_down_data)
        
        # Final summary
        if tc_result is not None:
            logging.info(f"FINAL SUMMARY - Critical temperature (overall): Tc = {tc_result:.3f} K")
            
            if tc_up is not None and tc_down is not None:
                hysteresis = tc_up - tc_down
                logging.info(f"FINAL SUMMARY - Tc (ramp up): {tc_up:.3f} K")
                logging.info(f"FINAL SUMMARY - Tc (ramp down): {tc_down:.3f} K")
                logging.info(f"FINAL SUMMARY - Hysteresis: {hysteresis:.3f} K")
                
                if abs(hysteresis) > 0.01:
                    logging.info(f"FINAL SUMMARY - Significant hysteresis detected: {hysteresis:.3f} K")
                else:
                    logging.info("FINAL SUMMARY - No significant hysteresis detected")
            
            logging.info(f"FINAL SUMMARY - Analysis methods: {tc_methods}")
            logging.info(f"FINAL SUMMARY - Statistics: {tc_stats}")
        else:
            logging.warning("FINAL SUMMARY - Critical temperature analysis did not return a valid result")
        
    finally:
        # Shut down heater and sourcemeter
        logging.info("Shutting down instruments...")
        shutdown_heater(temp_controller, config['instrument']['output_channel'])
        try:
            sourcemeter.set_output(False)
            logging.info("✓ Sourcemeter output disabled")
        except:
            logging.warning("Could not disable sourcemeter output")
    
    logging.info("Test complete!")
    return results_df

if __name__ == "__main__":
    results = main()
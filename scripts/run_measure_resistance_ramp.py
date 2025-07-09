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

# Create organized output directory structure
base_output_dir = "data"
os.makedirs(base_output_dir, exist_ok=True)

# Create timestamped folder for this measurement run
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(base_output_dir, f"resistance_ramp_{run_timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Configure logging with run-specific log file
log_filename = os.path.join(output_dir, 'resistance_ramp.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)

# Log the output directory for user reference
logging.info(f"Output directory: {output_dir}")

def setup_sourcemeter(sourcemeter: Keithley2400, current_level=1e-6, nplc=10.0, filter_count=10):
    """Setup Keithley 2400 for low-noise 2-wire resistance measurement."""
    logging.info("Setting up Keithley 2400 sourcemeter for low-noise operation...")
    
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
        
        # Enhanced measurement settings for noise reduction
        # Set integration time (NPLC) for better noise rejection
        sourcemeter.set_measurement_time(nplc)  # Higher NPLC for lower noise
        logging.info(f"Integration time set to: {nplc} NPLC ({nplc*16.7:.1f} ms)")
        
        # Enable digital filter if available
        try:
            # Try to enable digital filter with moving average
            sourcemeter.write(f":SENS:VOLT:DFIL:STAT ON")  # Enable digital filter
            sourcemeter.write(f":SENS:VOLT:DFIL:COUN {filter_count}")  # Set filter count
            sourcemeter.write(f":SENS:VOLT:DFIL:TCON MOV")  # Moving average filter
            logging.info(f"Digital filter enabled: {filter_count}-point moving average")
        except:
            logging.warning("Digital filter not available or failed to configure")
        
        # Set auto-range for voltage measurements
        try:
            sourcemeter.write(":SENS:VOLT:RANG:AUTO ON")
            logging.info("Auto-range enabled for voltage measurements")
        except:
            logging.warning("Auto-range configuration failed")
        
        # Enable autozero for better accuracy
        try:
            sourcemeter.write(":SYST:AZER ON")
            logging.info("Autozero enabled for better accuracy")
        except:
            logging.warning("Autozero configuration failed")
        
        # Enable output but don't turn on yet
        logging.info("✓ Sourcemeter configured for low-noise 2-wire resistance measurement")
        return True
        
    except Exception as e:
        logging.error(f"✗ Failed to setup sourcemeter: {str(e)}")
        return False

def measure_resistance(sourcemeter, current_level=1e-6, num_averages=5, settling_time=0.1):
    """Measure voltage with averaging and calculate resistance with improved noise handling."""
    try:
        # Allow settling time for stable measurement
        if settling_time > 0:
            time.sleep(settling_time)
        
        # Take multiple measurements for averaging
        voltages = []
        for i in range(num_averages):
            try:
                voltage = sourcemeter.read_voltage()
                if voltage is not None and not np.isnan(voltage):
                    voltages.append(voltage)
                    if num_averages > 1:
                        time.sleep(0.01)  # Small delay between measurements
            except:
                continue
        
        if len(voltages) == 0:
            logging.warning("No valid voltage measurements obtained")
            return None, None
        
        # Calculate statistics
        voltages = np.array(voltages)
        
        # Remove outliers using interquartile range method
        if len(voltages) >= 3:
            q75, q25 = np.percentile(voltages, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            voltages_filtered = voltages[(voltages >= lower_bound) & (voltages <= upper_bound)]
            
            if len(voltages_filtered) > 0:
                voltages = voltages_filtered
        
        # Calculate final voltage (mean of filtered measurements)
        voltage_mean = np.mean(voltages)
        voltage_std = np.std(voltages) if len(voltages) > 1 else 0
        
        # Calculate resistance using Ohm's law (V = I*R)
        resistance = voltage_mean / current_level
        resistance_uncertainty = voltage_std / current_level if voltage_std > 0 else 0
        
        # Log measurement quality occasionally
        if len(voltages) != num_averages:
            logging.debug(f"Measurement quality: {len(voltages)}/{num_averages} valid readings, "
                         f"σ(V) = {voltage_std*1e6:.1f} µV, σ(R) = {resistance_uncertainty:.3f} Ω")
        
        return voltage_mean, resistance
        
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
    
    # Enhanced measurement parameters for noise reduction
    nplc = config['sourcemeter'].get('integration_time_plc', 10.0)  # Integration time
    filter_count = config['sourcemeter'].get('filter_count', 10)   # Digital filter count
    num_averages = config['sourcemeter'].get('num_averages', 5)    # Number of measurements to average
    settling_time = config['sourcemeter'].get('settling_time', 0.1) # Settling time between measurements
    
    # Setup temperature controller
    setup_controller(temp_controller, output, config)
    
    # Setup sourcemeter with enhanced noise reduction
    if not setup_sourcemeter(sourcemeter, current_level, nplc, filter_count):
        raise RuntimeError("Failed to setup sourcemeter")
    
    # Set initial setpoint without ramping
    logging.info(f"Setting initial setpoint to {start_temp}K (no ramp)")
    configure_ramp(temp_controller, output, ramp_rate, enable=False, config=config)
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
                configure_ramp(temp_controller, output, ramp_rate, enable=True, config=config)
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
                configure_ramp(temp_controller, output, ramp_rate, enable=True, config=config)
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
            
            # Record resistance data with enhanced measurement
            voltage, resistance = measure_resistance(sourcemeter, current_level, num_averages, settling_time)
            
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
    plot_filename = os.path.join(output_dir, f"{filename_prefix}_time_series.png")
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
    plot_filename = os.path.join(output_dir, f"{filename_prefix}_vs_temp.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    logging.info(f"R vs T plot saved: {plot_filename}")
    
    plt.show()
    return plot_filename

def analyze_critical_temperature(df):
    """Analyze resistance data to identify critical temperature (Tc) with improved noise handling."""
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
    
    # Apply smoothing to reduce noise before derivative calculation
    try:
        from scipy.signal import savgol_filter
        # Savitzky-Golay filter for smoothing while preserving features
        window_length = min(11, len(resistances) // 3)  # Adaptive window size
        if window_length % 2 == 0:
            window_length += 1  # Must be odd
        if window_length >= 3:
            resistances_smooth = savgol_filter(resistances, window_length, 2)
            logging.info(f"Applied Savitzky-Golay smoothing (window={window_length})")
        else:
            resistances_smooth = resistances
            logging.warning("Insufficient data for smoothing")
    except ImportError:
        # Fallback to simple moving average if scipy not available
        try:
            window = min(5, len(resistances) // 4)
            if window >= 2:
                resistances_smooth = np.convolve(resistances, np.ones(window)/window, mode='same')
                logging.info(f"Applied moving average smoothing (window={window})")
            else:
                resistances_smooth = resistances
        except:
            resistances_smooth = resistances
            logging.warning("No smoothing applied - using raw data")
    
    # Method 1: Enhanced derivative method with phase-aware analysis
    tc_derivative = None
    max_derivative_value = None
    min_derivative_value = None
    
    if len(temperatures) > 5:
        # Calculate derivative on smoothed data
        dR_dT = np.gradient(resistances_smooth, temperatures)
        
        # Check if we have phase information for splitting analysis
        has_phases = 'phase' in valid_data.columns
        
        if has_phases:
            # Phase-aware derivative analysis
            ramp_up_mask = valid_data['phase'] == 'ramp_up'
            ramp_down_mask = valid_data['phase'] == 'ramp_down'
            
            ramp_up_indices = np.where(ramp_up_mask)[0]
            ramp_down_indices = np.where(ramp_down_mask)[0]
            
            tc_derivative_up = None
            tc_derivative_down = None
            
            # Analyze ramp up: look for maximum dR/dT (positive peak)
            if len(ramp_up_indices) > 3:
                dR_dT_up = dR_dT[ramp_up_indices]
                temp_up = temperatures[ramp_up_indices]
                
                try:
                    from scipy.signal import find_peaks
                    min_prominence = np.std(dR_dT_up) * 1.5
                    peaks, properties = find_peaks(dR_dT_up, prominence=min_prominence, distance=2)
                    
                    if len(peaks) > 0:
                        max_peak_idx = peaks[np.argmax(properties['prominences'])]
                        tc_derivative_up = temp_up[max_peak_idx]
                        max_derivative_value = dR_dT_up[max_peak_idx]
                        logging.info(f"Ramp UP - dR/dT maximum: Tc = {tc_derivative_up:.3f} K, "
                                   f"dR/dT = {max_derivative_value:.2e} Ω/K")
                    else:
                        max_idx = np.argmax(dR_dT_up)
                        tc_derivative_up = temp_up[max_idx]
                        max_derivative_value = dR_dT_up[max_idx]
                        logging.info(f"Ramp UP - dR/dT maximum: Tc = {tc_derivative_up:.3f} K, "
                                   f"dR/dT = {max_derivative_value:.2e} Ω/K (simple max)")
                except ImportError:
                    max_idx = np.argmax(dR_dT_up)
                    tc_derivative_up = temp_up[max_idx]
                    max_derivative_value = dR_dT_up[max_idx]
                    logging.info(f"Ramp UP - dR/dT maximum: Tc = {tc_derivative_up:.3f} K, "
                               f"dR/dT = {max_derivative_value:.2e} Ω/K")
            
            # Analyze ramp down: look for maximum dR/dT (positive peak)
            if len(ramp_down_indices) > 3:
                dR_dT_down = dR_dT[ramp_down_indices]
                temp_down = temperatures[ramp_down_indices]
                
                try:
                    from scipy.signal import find_peaks
                    min_prominence = np.std(dR_dT_down) * 1.5
                    peaks, properties = find_peaks(dR_dT_down, prominence=min_prominence, distance=2)
                    
                    if len(peaks) > 0:
                        max_peak_idx = peaks[np.argmax(properties['prominences'])]
                        tc_derivative_down = temp_down[max_peak_idx]
                        max_derivative_value_down = dR_dT_down[max_peak_idx]
                        logging.info(f"Ramp DOWN - dR/dT maximum: Tc = {tc_derivative_down:.3f} K, "
                                   f"dR/dT = {max_derivative_value_down:.2e} Ω/K")
                    else:
                        max_idx = np.argmax(dR_dT_down)
                        tc_derivative_down = temp_down[max_idx]
                        max_derivative_value_down = dR_dT_down[max_idx]
                        logging.info(f"Ramp DOWN - dR/dT maximum: Tc = {tc_derivative_down:.3f} K, "
                                   f"dR/dT = {max_derivative_value_down:.2e} Ω/K (simple max)")
                except ImportError:
                    max_idx = np.argmax(dR_dT_down)
                    tc_derivative_down = temp_down[max_idx]
                    max_derivative_value_down = dR_dT_down[max_idx]
                    logging.info(f"Ramp DOWN - dR/dT maximum: Tc = {tc_derivative_down:.3f} K, "
                               f"dR/dT = {max_derivative_value_down:.2e} Ω/K")
            
            # Choose the best derivative estimate (prefer ramp up if available)
            if tc_derivative_up is not None:
                tc_derivative = tc_derivative_up
                logging.info(f"Using ramp UP derivative for overall Tc estimate: {tc_derivative:.3f} K")
            elif tc_derivative_down is not None:
                tc_derivative = tc_derivative_down
                logging.info(f"Using ramp DOWN derivative for overall Tc estimate: {tc_derivative:.3f} K")
        
        else:
            # No phase information - use original method on full dataset
            try:
                from scipy.signal import find_peaks
                min_prominence = np.std(dR_dT) * 2
                peaks, properties = find_peaks(dR_dT, prominence=min_prominence, distance=3)
                
                if len(peaks) > 0:
                    max_peak_idx = peaks[np.argmax(properties['prominences'])]
                    tc_derivative = temperatures[max_peak_idx]
                    max_derivative_value = dR_dT[max_peak_idx]
                    logging.info(f"Method 1 (Enhanced dR/dT): Tc = {tc_derivative:.3f} K, "
                               f"dR/dT = {max_derivative_value:.2e} Ω/K (prominence method)")
                else:
                    max_derivative_idx = np.argmax(dR_dT)
                    tc_derivative = temperatures[max_derivative_idx]
                    max_derivative_value = dR_dT[max_derivative_idx]
                    logging.info(f"Method 1 (Enhanced dR/dT): Tc = {tc_derivative:.3f} K, "
                               f"dR/dT = {max_derivative_value:.2e} Ω/K (simple maximum)")
            except ImportError:
                max_derivative_idx = np.argmax(dR_dT)
                tc_derivative = temperatures[max_derivative_idx]
                max_derivative_value = dR_dT[max_derivative_idx]
                logging.info(f"Method 1 (Enhanced dR/dT): Tc = {tc_derivative:.3f} K, "
                           f"dR/dT = {max_derivative_value:.2e} Ω/K")
    else:
        logging.warning("Insufficient data points for derivative method")
    
    # Method 2: Improved threshold method with adaptive thresholds
    # Use robust statistics to define thresholds
    low_percentile = np.percentile(resistances_smooth, 10)  # 10th percentile (SC state)
    high_percentile = np.percentile(resistances_smooth, 90) # 90th percentile (normal state)
    resistance_range = high_percentile - low_percentile
    
    # Use 10% of the resistance range above low percentile as threshold
    threshold_resistance = low_percentile + 0.1 * resistance_range
    
    # Find first temperature where smoothed resistance exceeds threshold
    above_threshold = resistances_smooth > threshold_resistance
    if np.any(above_threshold):
        threshold_idx = np.where(above_threshold)[0][0]
        tc_threshold = temperatures[threshold_idx]
        logging.info(f"Method 2 (Improved threshold): Tc = {tc_threshold:.3f} K, "
                   f"R_threshold = {threshold_resistance:.3f} Ω")
    else:
        tc_threshold = None
        logging.warning("No data points above resistance threshold")
    
    # Method 3: Enhanced midpoint method
    midpoint_resistance = low_percentile + 0.5 * resistance_range
    
    # Find temperature closest to midpoint resistance using smoothed data
    midpoint_diff = np.abs(resistances_smooth - midpoint_resistance)
    midpoint_idx = np.argmin(midpoint_diff)
    tc_midpoint = temperatures[midpoint_idx]
    
    logging.info(f"Method 3 (Enhanced midpoint): Tc = {tc_midpoint:.3f} K, "
               f"R_midpoint = {midpoint_resistance:.3f} Ω")
    
    # Method 4: Improved onset method with better region detection
    try:
        # Use derivative information to better define regions
        if tc_derivative is not None:
            # Find regions based on derivative behavior
            derivative_threshold = max_derivative_value * 0.1 if max_derivative_value else 0
            
            # Find superconducting region (before significant dR/dT)
            sc_region = dR_dT < derivative_threshold
            if np.any(sc_region):
                sc_end_idx = np.where(~sc_region)[0][0] if np.any(~sc_region) else len(temperatures)//3
            else:
                sc_end_idx = len(temperatures) // 3
            
            # Find normal region (after significant dR/dT)
            normal_region = dR_dT < derivative_threshold
            normal_indices = np.where(normal_region)[0]
            if len(normal_indices) > 0 and normal_indices[-1] > sc_end_idx:
                normal_start_idx = normal_indices[-1]
            else:
                normal_start_idx = min(len(temperatures) - 3, int(0.7 * len(temperatures)))
            
            if sc_end_idx < normal_start_idx and sc_end_idx >= 3 and normal_start_idx < len(temperatures) - 3:
                # Fit lines to superconducting and normal regions
                sc_temps = temperatures[:sc_end_idx]
                sc_resistances = resistances_smooth[:sc_end_idx]
                normal_temps = temperatures[normal_start_idx:]
                normal_resistances = resistances_smooth[normal_start_idx:]
                
                if len(sc_temps) >= 3 and len(normal_temps) >= 3:
                    # Linear fits with error handling
                    sc_fit = np.polyfit(sc_temps, sc_resistances, 1)
                    normal_fit = np.polyfit(normal_temps, normal_resistances, 1)
                    
                    # Find intersection (onset Tc)
                    if abs(sc_fit[0] - normal_fit[0]) > 1e-10:
                        tc_onset = (normal_fit[1] - sc_fit[1]) / (sc_fit[0] - normal_fit[0])
                        
                        # Check if intersection is within reasonable range
                        if temperatures[0] <= tc_onset <= temperatures[-1]:
                            logging.info(f"Method 4 (Improved onset): Tc = {tc_onset:.3f} K")
                        else:
                            tc_onset = None
                            logging.warning("Improved onset method: intersection outside temperature range")
                    else:
                        tc_onset = None
                        logging.warning("Improved onset method: parallel lines, no intersection")
                else:
                    tc_onset = None
                    logging.warning("Improved onset method: insufficient data points in regions")
            else:
                tc_onset = None
                logging.warning("Improved onset method: could not define proper regions")
        else:
            tc_onset = None
            logging.warning("Improved onset method: no derivative data available")
    except Exception as e:
        tc_onset = None
        logging.warning(f"Improved onset method failed: {str(e)}")
    
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
        threshold_90pct = low_percentile + 0.9 * resistance_range
        above_90pct = resistances_smooth > threshold_90pct
        if np.any(above_90pct):
            threshold_90_idx = np.where(above_90pct)[0][0]
            tc_90pct = temperatures[threshold_90_idx]
            temp_transition_width = tc_90pct - tc_threshold
            logging.info(f"Temperature transition width (10%-90%): {temp_transition_width:.3f} K")
        else:
            logging.warning("Could not find 90% threshold for transition width calculation")
    
    # Calculate statistics with improved weighting
    valid_tc_values = [tc for tc in tc_methods.values() if tc is not None]
    if valid_tc_values:
        mean_tc = np.mean(valid_tc_values)
        std_tc = np.std(valid_tc_values) if len(valid_tc_values) > 1 else 0
        
        logging.info("CRITICAL TEMPERATURE SUMMARY:")
        logging.info(f"Available methods: {len(valid_tc_values)}")
        logging.info(f"Mean Tc: {mean_tc:.3f} ± {std_tc:.3f} K")
        logging.info(f"Resistance range: {low_percentile:.3f} - {high_percentile:.3f} Ω")
        if temp_transition_width is not None:
            logging.info(f"Temperature transition width (10%-90%): {temp_transition_width:.3f} K")
        
        # Choose best estimate with improved priority (derivative method preferred if reliable)
        if tc_derivative is not None and std_tc < 0.1:  # Derivative method if consistent with others
            best_tc = tc_derivative
            best_method = "derivative"
        elif tc_midpoint is not None:
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
            'min_resistance': low_percentile,
            'max_resistance': high_percentile,
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
        method_names = {'derivative': 'dR/dT maximum', 'threshold_10pct': '10% threshold', 
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
    
    # Plot 2: Log scale R vs T with separate up/down sweeps
    for phase in valid_data['phase'].unique():
        phase_data = valid_data[valid_data['phase'] == phase]
        if len(phase_data) > 0:
            ax2.semilogy(phase_data['actual_temp_K'], phase_data['resistance_ohm'], 
                        color=phase_colors.get(phase, 'black'), linewidth=2, 
                        label=f'{phase} phase', alpha=0.8)
    
    if best_tc is not None:
        ax2.axvline(x=best_tc, color='red', linestyle='--', linewidth=2, 
                   label=f'Best Tc = {best_tc:.3f} K')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Resistance (Ω) [log scale]')
    ax2.set_title('Resistance vs Temperature (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: dR/dT vs T with phase separation
    if len(temperatures) > 5:
        # Plot dR/dT for each phase separately
        for phase in valid_data['phase'].unique():
            phase_data = valid_data[valid_data['phase'] == phase]
            if len(phase_data) > 3:
                phase_temps = phase_data['actual_temp_K'].values
                phase_resistances = phase_data['resistance_ohm'].values
                phase_dR_dT = np.gradient(phase_resistances, phase_temps)
                
                ax3.plot(phase_temps, phase_dR_dT, color=phase_colors.get(phase, 'black'), 
                        linewidth=2, label=f'dR/dT ({phase})', alpha=0.8)
                
                # Mark extrema for ramp phases
                if phase == 'ramp_up' and len(phase_dR_dT) > 0:
                    max_idx = np.argmax(phase_dR_dT)
                    ax3.plot(phase_temps[max_idx], phase_dR_dT[max_idx], 'bo', markersize=8, 
                            label=f'Max dR/dT (up): {phase_temps[max_idx]:.3f} K')
                    ax3.axvline(x=phase_temps[max_idx], color='blue', linestyle=':', alpha=0.7)
                
                elif phase == 'ramp_down' and len(phase_dR_dT) > 0:
                    max_idx = np.argmax(phase_dR_dT)
                    ax3.plot(phase_temps[max_idx], phase_dR_dT[max_idx], 'ro', markersize=8, 
                            label=f'Max dR/dT (down): {phase_temps[max_idx]:.3f} K')
                    ax3.axvline(x=phase_temps[max_idx], color='red', linestyle=':', alpha=0.7)
        
        ax3.set_xlabel('Temperature (K)')
        ax3.set_ylabel('dR/dT (Ω/K)')
        ax3.set_title('Temperature Derivative of Resistance (Phase-Aware)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Normalized resistance showing transition with separate up/down sweeps
    if tc_stats is not None:
        min_r = tc_stats['min_resistance']
        max_r = tc_stats['max_resistance']
        
        for phase in valid_data['phase'].unique():
            phase_data = valid_data[valid_data['phase'] == phase]
            if len(phase_data) > 0:
                phase_resistances = phase_data['resistance_ohm'].values
                normalized_r = (phase_resistances - min_r) / (max_r - min_r)
                ax4.plot(phase_data['actual_temp_K'], normalized_r, 
                        color=phase_colors.get(phase, 'black'), linewidth=2, 
                        label=f'{phase} phase', alpha=0.8)
        
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
    plot_filename = os.path.join(output_dir, f"{filename_prefix}_tc_analysis.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    logging.info(f"Tc analysis plot saved: {plot_filename}")
    
    plt.show()
    return plot_filename

def filter_drdt(dR_dT, temperatures, filter_config):
    """
    Apply configurable filtering to dR/dT data to reduce noise.
    
    Args:
        dR_dT: Array of dR/dT values
        temperatures: Array of temperature values (for filtering validation)
        filter_config: Dictionary with filtering configuration
    
    Returns:
        filtered_dR_dT: Filtered dR/dT array
    """
    if not filter_config.get('enabled', True):
        logging.info("dR/dT filtering disabled")
        return dR_dT
    
    if len(dR_dT) < 3:
        logging.warning("Insufficient data for dR/dT filtering")
        return dR_dT
    
    method = filter_config.get('method', 'savgol')
    filter_twice = filter_config.get('filter_twice', False)
    preserve_peaks = filter_config.get('preserve_peaks', True)
    
    filtered_dR_dT = dR_dT.copy()
    
    try:
        if method == 'savgol':
            # Savitzky-Golay filter
            window = filter_config.get('savgol_window', 5)
            order = filter_config.get('savgol_order', 2)
            
            # Ensure window is odd and reasonable
            if window % 2 == 0:
                window += 1
            window = max(3, min(window, len(dR_dT)))
            if window % 2 == 0:
                window -= 1
            
            # Ensure order is less than window
            order = min(order, window - 1)
            
            try:
                from scipy.signal import savgol_filter
                filtered_dR_dT = savgol_filter(filtered_dR_dT, window, order)
                logging.info(f"Applied Savitzky-Golay filter to dR/dT (window={window}, order={order})")
            except ImportError:
                logging.warning("SciPy not available for Savitzky-Golay filtering, using moving average")
                method = 'moving_average'
        
        if method == 'moving_average':
            # Simple moving average
            window = filter_config.get('moving_avg_window', 5)
            window = max(2, min(window, len(dR_dT)))
            
            filtered_dR_dT = np.convolve(filtered_dR_dT, np.ones(window)/window, mode='same')
            logging.info(f"Applied moving average filter to dR/dT (window={window})")
        
        elif method == 'gaussian':
            # Gaussian filter
            sigma = filter_config.get('gaussian_sigma', 1.0)
            
            try:
                from scipy.ndimage import gaussian_filter1d
                filtered_dR_dT = gaussian_filter1d(filtered_dR_dT, sigma)
                logging.info(f"Applied Gaussian filter to dR/dT (sigma={sigma})")
            except ImportError:
                logging.warning("SciPy not available for Gaussian filtering, using moving average")
                window = max(2, min(5, len(dR_dT)))
                filtered_dR_dT = np.convolve(filtered_dR_dT, np.ones(window)/window, mode='same')
        
        elif method == 'median':
            # Median filter
            window = filter_config.get('median_window', 5)
            if window % 2 == 0:
                window += 1
            window = max(3, min(window, len(dR_dT)))
            
            try:
                from scipy.signal import medfilt
                filtered_dR_dT = medfilt(filtered_dR_dT, window)
                logging.info(f"Applied median filter to dR/dT (window={window})")
            except ImportError:
                logging.warning("SciPy not available for median filtering, using moving average")
                window = max(2, min(5, len(dR_dT)))
                filtered_dR_dT = np.convolve(filtered_dR_dT, np.ones(window)/window, mode='same')
        
        else:
            logging.warning(f"Unknown filtering method '{method}', using raw dR/dT")
            return dR_dT
        
        # Apply filtering twice if requested
        if filter_twice:
            if method == 'savgol':
                try:
                    from scipy.signal import savgol_filter
                    filtered_dR_dT = savgol_filter(filtered_dR_dT, window, order)
                    logging.info("Applied second pass of Savitzky-Golay filter")
                except ImportError:
                    pass
            elif method == 'moving_average':
                filtered_dR_dT = np.convolve(filtered_dR_dT, np.ones(window)/window, mode='same')
                logging.info("Applied second pass of moving average filter")
            elif method == 'gaussian':
                try:
                    from scipy.ndimage import gaussian_filter1d
                    filtered_dR_dT = gaussian_filter1d(filtered_dR_dT, sigma)
                    logging.info("Applied second pass of Gaussian filter")
                except ImportError:
                    pass
            elif method == 'median':
                try:
                    from scipy.signal import medfilt
                    filtered_dR_dT = medfilt(filtered_dR_dT, window)
                    logging.info("Applied second pass of median filter")
                except ImportError:
                    pass
        
        # Log filtering effectiveness
        original_std = np.std(dR_dT)
        filtered_std = np.std(filtered_dR_dT)
        noise_reduction = (1 - filtered_std / original_std) * 100 if original_std > 0 else 0
        logging.info(f"dR/dT filtering effectiveness: {noise_reduction:.1f}% noise reduction")
        
        return filtered_dR_dT
    
    except Exception as e:
        logging.error(f"Error in dR/dT filtering: {str(e)}")
        return dR_dT


def save_analysis_results(tc_result, tc_methods, tc_stats, tc_up, tc_down, 
                         ramp_up_data, ramp_down_data, output_dir, filename_prefix):
    """
    Save analysis results to a CSV file.
    
    Args:
        tc_result: Overall critical temperature
        tc_methods: Dictionary of Tc values from different methods
        tc_stats: Dictionary of statistics from analysis
        tc_up: Critical temperature from ramp up data
        tc_down: Critical temperature from ramp down data
        ramp_up_data: DataFrame of ramp up data
        ramp_down_data: DataFrame of ramp down data
        output_dir: Directory to save the file
        filename_prefix: Prefix for the filename
    """
    try:
        # Create timestamp for unique filename
        analysis_filename = os.path.join(output_dir, f"{filename_prefix}_analysis.csv")
        
        # Prepare analysis data
        analysis_data = []
        
        # Overall results
        analysis_data.append({
            'Parameter': 'Overall_Tc_K',
            'Value': tc_result if tc_result is not None else 'N/A',
            'Method': 'Best estimate',
            'Description': 'Critical temperature from overall dataset'
        })
        
        # Individual method results
        if tc_methods:
            for method, value in tc_methods.items():
                analysis_data.append({
                    'Parameter': f'Tc_{method}_K',
                    'Value': value if value is not None else 'N/A',
                    'Method': method,
                    'Description': f'Critical temperature using {method} method'
                })
        
        # Statistics
        if tc_stats:
            for stat, value in tc_stats.items():
                if stat in ['min_resistance', 'max_resistance', 'transition_width_K']:
                    unit = 'Ohm' if 'resistance' in stat else 'K'
                    analysis_data.append({
                        'Parameter': stat,
                        'Value': value if value is not None else 'N/A',
                        'Method': 'Statistical analysis',
                        'Description': f'{stat.replace("_", " ").title()} ({unit})'
                    })
        
        # Ramp-specific results
        if tc_up is not None:
            analysis_data.append({
                'Parameter': 'Tc_ramp_up_K',
                'Value': tc_up,
                'Method': 'Ramp up analysis',
                'Description': 'Critical temperature during temperature ramp up'
            })
        
        if tc_down is not None:
            analysis_data.append({
                'Parameter': 'Tc_ramp_down_K',
                'Value': tc_down,
                'Method': 'Ramp down analysis',
                'Description': 'Critical temperature during temperature ramp down'
            })
        
        # Hysteresis calculation
        if tc_up is not None and tc_down is not None:
            hysteresis = tc_up - tc_down
            analysis_data.append({
                'Parameter': 'Hysteresis_K',
                'Value': hysteresis,
                'Method': 'Difference calculation',
                'Description': 'Temperature hysteresis (Tc_up - Tc_down)'
            })
            
            analysis_data.append({
                'Parameter': 'Significant_hysteresis',
                'Value': 'Yes' if abs(hysteresis) > 0.01 else 'No',
                'Method': 'Threshold check',
                'Description': 'Significant hysteresis (>0.01 K)'
            })
        
        # Data quality metrics
        total_points = len(ramp_up_data) + len(ramp_down_data) if ramp_up_data is not None and ramp_down_data is not None else 'N/A'
        analysis_data.append({
            'Parameter': 'Total_data_points',
            'Value': total_points,
            'Method': 'Count',
            'Description': 'Total number of data points collected'
        })
        
        if ramp_up_data is not None:
            analysis_data.append({
                'Parameter': 'Ramp_up_points',
                'Value': len(ramp_up_data),
                'Method': 'Count',
                'Description': 'Number of data points during ramp up'
            })
        
        if ramp_down_data is not None:
            analysis_data.append({
                'Parameter': 'Ramp_down_points',
                'Value': len(ramp_down_data),
                'Method': 'Count',
                'Description': 'Number of data points during ramp down'
            })
        
        # Convert to DataFrame and save
        analysis_df = pd.DataFrame(analysis_data)
        analysis_df.to_csv(analysis_filename, index=False)
        
        logging.info(f"✓ Analysis results saved to: {analysis_filename}")
        return analysis_filename
        
    except Exception as e:
        logging.error(f"Failed to save analysis results: {str(e)}")
        return None

def create_measurement_summary(config, tc_result, tc_methods, tc_stats, tc_up, tc_down, 
                              ramp_up_data, ramp_down_data, output_dir):
    """Create a summary file with measurement configuration and key results."""
    try:
        summary_filename = os.path.join(output_dir, "measurement_summary.txt")
        
        with open(summary_filename, 'w') as f:
            f.write("RESISTANCE RAMP MEASUREMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Timestamp and run info
            f.write(f"Measurement Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {output_dir}\n\n")
            
            # Configuration summary
            f.write("MEASUREMENT CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            
            if config:
                # Temperature controller settings
                instrument_config = config.get('instrument', {})
                f.write(f"Temperature Controller: {instrument_config.get('port', 'N/A')}\n")
                f.write(f"Output Channel: {instrument_config.get('output_channel', 'N/A')}\n")
                
                # Sourcemeter settings
                sourcemeter_config = config.get('sourcemeter', {})
                f.write(f"Sourcemeter: {sourcemeter_config.get('port', 'N/A')}\n")
                f.write(f"Source Current: {sourcemeter_config.get('current_level', 1e-6)*1e6:.1f} µA\n")
                f.write(f"Integration Time: {sourcemeter_config.get('integration_time_plc', 10.0)} NPLC\n")
                f.write(f"Digital Filter: {sourcemeter_config.get('filter_count', 10)} points\n")
                f.write(f"Measurement Averages: {sourcemeter_config.get('num_averages', 5)}\n")
                
                # Ramp test parameters
                ramp_config = config.get('ramp_test', {})
                f.write(f"Start Temperature: {ramp_config.get('start_temp', 'N/A')} K\n")
                f.write(f"Target Temperature: {ramp_config.get('target_temp', 'N/A')} K\n")
                f.write(f"Ramp Rate: {ramp_config.get('ramp_rate', 'N/A')} K/min\n")
                f.write(f"Hold Time: {ramp_config.get('hold_time', 'N/A')} s\n")
                f.write(f"Recording Interval: {ramp_config.get('recording_interval', 'N/A')} s\n")
                
                # PID settings
                pid_config = config.get('pid', {})
                f.write(f"PID Settings: P={pid_config.get('P', 'N/A')}, I={pid_config.get('I', 'N/A')}, D={pid_config.get('D', 'N/A')}\n")
            
            f.write("\n")
            
            # Critical temperature results
            f.write("CRITICAL TEMPERATURE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            if tc_result is not None:
                f.write(f"Best Tc Estimate: {tc_result:.3f} K ({tc_stats.get('best_method', 'N/A')})\n")
                
                if tc_methods:
                    f.write("Individual Method Results:\n")
                    method_names = {'derivative': 'dR/dT maximum', 'threshold_10pct': '10% threshold', 
                                   'midpoint_50pct': '50% midpoint', 'linear_onset': 'linear onset'}
                    for method, value in tc_methods.items():
                        if value is not None:
                            f.write(f"  {method_names.get(method, method)}: {value:.3f} K\n")
                
                if tc_stats:
                    f.write(f"Statistics: Mean = {tc_stats.get('mean_tc', 'N/A'):.3f} ± {tc_stats.get('std_tc', 'N/A'):.3f} K\n")
                    f.write(f"Resistance Range: {tc_stats.get('min_resistance', 'N/A'):.3f} - {tc_stats.get('max_resistance', 'N/A'):.3f} Ω\n")
                    if tc_stats.get('temp_transition_width'):
                        f.write(f"Transition Width: {tc_stats['temp_transition_width']:.3f} K\n")
            else:
                f.write("No valid critical temperature found\n")
            
            f.write("\n")
            
            # Hysteresis analysis
            f.write("HYSTERESIS ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            if tc_up is not None and tc_down is not None:
                hysteresis = tc_up - tc_down
                f.write(f"Tc (ramp up): {tc_up:.3f} K\n")
                f.write(f"Tc (ramp down): {tc_down:.3f} K\n")
                f.write(f"Hysteresis: {hysteresis:.3f} K\n")
                if abs(hysteresis) > 0.01:
                    f.write("Significant hysteresis detected (>0.01 K)\n")
                else:
                    f.write("No significant hysteresis detected\n")
            elif tc_up is not None:
                f.write(f"Tc (ramp up only): {tc_up:.3f} K\n")
                f.write("Insufficient ramp down data for hysteresis analysis\n")
            elif tc_down is not None:
                f.write(f"Tc (ramp down only): {tc_down:.3f} K\n")
                f.write("Insufficient ramp up data for hysteresis analysis\n")
            else:
                f.write("Insufficient data for hysteresis analysis\n")
            
            f.write("\n")
            
            # Data quality summary
            f.write("DATA QUALITY:\n")
            f.write("-" * 30 + "\n")
            
            if ramp_up_data is not None:
                f.write(f"Ramp Up Data Points: {len(ramp_up_data)}\n")
            if ramp_down_data is not None:
                f.write(f"Ramp Down Data Points: {len(ramp_down_data)}\n")
            
            f.write("\n")
            
            # Output files
            f.write("OUTPUT FILES:\n")
            f.write("-" * 30 + "\n")
            f.write("resistance_ramp_data.csv - Raw measurement data\n")
            f.write("resistance_ramp_analysis.csv - Analysis results\n")
            f.write("resistance_ramp_time_series.png - Time series plots\n")
            f.write("resistance_vs_temp.png - R vs T plot with Tc analysis\n")
            f.write("resistance_tc_analysis.png - Detailed Tc analysis plots\n")
            f.write("resistance_ramp.log - Detailed log file\n")
            f.write("measurement_summary.txt - This summary file\n")
        
        logging.info(f"✓ Measurement summary saved to: {summary_filename}")
        return summary_filename
        
    except Exception as e:
        logging.error(f"Failed to create measurement summary: {str(e)}")
        return None

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
        
        # Save raw data
        filename = os.path.join(output_dir, "resistance_ramp_data.csv")
        results_df.to_csv(filename, index=False)
        logging.info(f"✓ Raw sweep data saved to: {filename}")
        
        # Create plots
        create_resistance_ramp_plot(results_df, "resistance_ramp", config, output_dir)
        create_resistance_vs_temperature_plot(results_df, "resistance", config, output_dir)
        create_critical_temperature_analysis_plot(results_df, "resistance", config, output_dir)
        
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
        
        # Save analysis results to CSV
        save_analysis_results(tc_result, tc_methods, tc_stats, tc_up, tc_down, 
                             ramp_up_data, ramp_down_data, output_dir, "resistance_ramp")
        
        # Create measurement summary
        create_measurement_summary(config, tc_result, tc_methods, tc_stats, tc_up, tc_down, 
                                  ramp_up_data, ramp_down_data, output_dir)
        
        logging.info("=" * 60)
        logging.info(f"✓ All results saved to: {output_dir}")
        logging.info("=" * 60)
        
        # Create measurement summary
        create_measurement_summary(config, tc_result, tc_methods, tc_stats, tc_up, tc_down, 
                                  ramp_up_data, ramp_down_data, output_dir)
        
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
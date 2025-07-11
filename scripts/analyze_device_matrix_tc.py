#!/usr/bin/env python3
"""
Device Matrix Tc Analysis Script

This script analyzes resistance ramp measurements across the entire device matrix (A1-G7)
and creates comprehensive visualizations of critical temperature distribution.

Author: Lab Analysis Script
Date: July 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import glob
import re
from pathlib import Path
import warnings
import yaml
from scipy.io import loadmat
warnings.filterwarnings('ignore')

# Import the critical temperature analysis function from the measurement script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_file="configs/device_matrix_analysis_config.yaml"):
    """Load configuration from YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), config_file)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return None

# Load configuration
CONFIG = load_config()

# Set up output directory early for logging
output_base_dir = CONFIG['output']['base_directory'] if CONFIG else "device_matrix_analysis_results"
os.makedirs(output_base_dir, exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "data"), exist_ok=True)

# Set up logging based on config
log_level = getattr(logging, CONFIG['logging']['level'] if CONFIG else 'INFO')
log_handlers = [logging.StreamHandler()]

if CONFIG is None or CONFIG['logging']['save_log']:
    log_filename = f'device_matrix_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_filepath = os.path.join(output_base_dir, "data", log_filename)
    log_handlers.append(logging.FileHandler(log_filepath))

logging.basicConfig(
    level=log_level,
    format=CONFIG['logging']['log_format'] if CONFIG else '%(asctime)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)

# Log the log file location
if len(log_handlers) > 1:  # More than just console handler
    logging.info(f"Log file will be saved to: {log_filepath}")

# Configuration with defaults
if CONFIG:
    BASE_DATA_PATH = CONFIG['data_source']['base_path']
    OUTPUT_DIR = CONFIG['output']['base_directory']
    MEASUREMENT_TYPE = CONFIG['data_source']['measurement_type']
    ROWS = CONFIG['matrix']['rows']
    COLS = CONFIG['matrix']['columns']
else:
    # Default configuration
    BASE_DATA_PATH = r"S:\SC\Measurements\SPG806\C6"
    OUTPUT_DIR = "device_matrix_analysis_results"
    MEASUREMENT_TYPE = "resistance_ramp"
    ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    COLS = ['1', '2', '3', '4', '5', '6', '7']

def create_output_directory():
    """Create output directory without timestamp - plots will be updated on each run."""
    output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Using output directory: {output_dir}")
    
    # Create subdirectories for different types of outputs
    plots_dir = os.path.join(output_dir, "plots")
    data_dir = os.path.join(output_dir, "data")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    logging.info(f"Plots will be saved to: {plots_dir}")
    logging.info(f"Data files will be saved to: {data_dir}")
    
    return output_dir

def analyze_critical_temperature_simple(df):
    """
    Simplified version of the critical temperature analysis function from the main script.
    Returns just the best Tc estimate and basic statistics.
    """
    # Get configuration parameters
    min_points = CONFIG['analysis']['min_data_points'] if CONFIG else 10
    
    # Get valid resistance data
    valid_data = df.dropna(subset=['resistance_ohm', 'actual_temp_K']).copy()
    if len(valid_data) < min_points:
        return None, None, None
    
    # Sort by temperature for proper analysis
    valid_data = valid_data.sort_values('actual_temp_K')
    temperatures = valid_data['actual_temp_K'].values
    resistances = valid_data['resistance_ohm'].values
    
    # Apply smoothing based on configuration
    smoothing_enabled = CONFIG['analysis']['smoothing_enabled'] if CONFIG else True
    smoothing_method = CONFIG['analysis']['smoothing_method'] if CONFIG else 'savgol'
    
    if smoothing_enabled:
        try:
            if smoothing_method == 'savgol':
                from scipy.signal import savgol_filter
                window_length = CONFIG['analysis']['smoothing_window'] if CONFIG else 11
                window_length = min(window_length, len(resistances) // 3)
                if window_length % 2 == 0:
                    window_length += 1
                if window_length >= 3:
                    resistances_smooth = savgol_filter(resistances, window_length, 2)
                else:
                    resistances_smooth = resistances
            else:
                # Simple moving average fallback
                window = min(5, len(resistances) // 4)
                if window >= 2:
                    resistances_smooth = np.convolve(resistances, np.ones(window)/window, mode='same')
                else:
                    resistances_smooth = resistances
        except ImportError:
            # Simple moving average fallback
            window = min(5, len(resistances) // 4)
            if window >= 2:
                resistances_smooth = np.convolve(resistances, np.ones(window)/window, mode='same')
            else:
                resistances_smooth = resistances
    else:
        resistances_smooth = resistances
    
    # Method 1: Derivative method
    tc_derivative = None
    if len(temperatures) > 5:
        dR_dT = np.gradient(resistances_smooth, temperatures)
        max_derivative_idx = np.argmax(dR_dT)
        tc_derivative = temperatures[max_derivative_idx]
    
    # Method 2: Threshold method (10% above minimum)
    low_percentile = np.percentile(resistances_smooth, 10)
    high_percentile = np.percentile(resistances_smooth, 90)
    resistance_range = high_percentile - low_percentile
    threshold_resistance = low_percentile + 0.1 * resistance_range
    
    above_threshold = resistances_smooth > threshold_resistance
    tc_threshold = None
    if np.any(above_threshold):
        threshold_idx = np.where(above_threshold)[0][0]
        tc_threshold = temperatures[threshold_idx]
    
    # Method 3: Midpoint method (50% of transition)
    midpoint_resistance = low_percentile + 0.5 * resistance_range
    midpoint_diff = np.abs(resistances_smooth - midpoint_resistance)
    midpoint_idx = np.argmin(midpoint_diff)
    tc_midpoint = temperatures[midpoint_idx]
    
    # Choose best estimate with configurable thresholds
    tc_methods = {
        'derivative': tc_derivative,
        'threshold': tc_threshold,
        'midpoint': tc_midpoint
    }
    
    valid_tc_values = [tc for tc in tc_methods.values() if tc is not None]
    if valid_tc_values:
        mean_tc = np.mean(valid_tc_values)
        std_tc = np.std(valid_tc_values) if len(valid_tc_values) > 1 else 0
        
        # Get quality thresholds from config
        good_threshold = CONFIG['quality']['good_std_threshold'] if CONFIG else 0.05
        
        # Choose best estimate (prefer derivative if available and consistent)
        if tc_derivative is not None and std_tc < good_threshold:
            best_tc = tc_derivative
            best_method = "derivative"
        elif tc_midpoint is not None:
            best_tc = tc_midpoint
            best_method = "midpoint"
        elif tc_threshold is not None:
            best_tc = tc_threshold
            best_method = "threshold"
        else:
            best_tc = tc_derivative
            best_method = "derivative"
        
        stats = {
            'min_resistance': low_percentile,
            'max_resistance': high_percentile,
            'resistance_range': resistance_range,
            'mean_tc': mean_tc,
            'std_tc': std_tc,
            'best_method': best_method,
            'num_methods': len(valid_tc_values)
        }
        
        return best_tc, tc_methods, stats
    else:
        return None, tc_methods, None

def find_device_data_files(base_path, device_name):
    """
    Find all data files for a specific device.
    Returns list of (file_path, measurement_date) tuples.
    """
    device_path = os.path.join(base_path, device_name)
    logging.debug(f"Searching for device {device_name} at path: {device_path}")
    
    if not os.path.exists(device_path):
        logging.debug(f"Device path does not exist: {device_path}")
        return []
    
    # Get file pattern from config
    file_pattern = CONFIG['file_patterns']['data_file_pattern'] if CONFIG else "*resistance_ramp*.mat"
    date_pattern = CONFIG['file_patterns']['date_extraction_pattern'] if CONFIG else r"(\d{8}_\d{6})"
    
    logging.debug(f"Using file pattern: {file_pattern}")
    logging.debug(f"Using date pattern: {date_pattern}")
    
    # Look for measurement directories and files more broadly
    measurement_dirs = []
    
    # First, check the standard structure: device_path/measurement_type/
    standard_measurement_dir = os.path.join(device_path, MEASUREMENT_TYPE)
    if os.path.exists(standard_measurement_dir):
        measurement_dirs.append(standard_measurement_dir)
        logging.debug(f"Found standard measurement directory: {standard_measurement_dir}")
    
    # Also check for other common measurement directory names
    common_measurement_dirs = [
        'resistance_ramp',
        'resistance_measurements', 
        'tc_measurements',
        'critical_temperature',
        'ramp_measurements'
    ]
    
    for dir_name in common_measurement_dirs:
        potential_dir = os.path.join(device_path, dir_name)
        if os.path.exists(potential_dir) and potential_dir not in measurement_dirs:
            measurement_dirs.append(potential_dir)
            logging.debug(f"Found measurement directory: {potential_dir}")
    
    # Check if there are MAT files directly in the device directory
    direct_mat_files = glob.glob(os.path.join(device_path, "*.mat"))
    if direct_mat_files:
        logging.debug(f"Found {len(direct_mat_files)} MAT files directly in device directory")
        measurement_dirs.append(device_path)
    
    # Search in all subdirectories for any directories that might contain resistance data
    for root, dirs, files in os.walk(device_path):
        logging.debug(f"Searching in directory: {root}")
        logging.debug(f"  Subdirectories: {dirs}")
        logging.debug(f"  Files: {files[:5]}...")  # Show first 5 files
        
        for dir_name in dirs:
            dir_lower = dir_name.lower()
            full_dir_path = os.path.join(root, dir_name)
            
            # Look for various patterns that might contain resistance data
            if any(pattern in dir_lower for pattern in [
                'resistance',
                'ramp',
                'tc',
                'critical',
                'measurement',
                'temp'
            ]) and full_dir_path not in measurement_dirs:
                measurement_dirs.append(full_dir_path)
                logging.debug(f"  Added measurement directory: {full_dir_path}")
        
        # Also check if current directory has relevant MAT files
        mat_files_in_dir = [f for f in files if f.lower().endswith('.mat')]
        if mat_files_in_dir and root != device_path and root not in measurement_dirs:
            has_resistance_data = any(any(keyword in f.lower() for keyword in ['resistance', 'ramp', 'tc', 'critical']) 
                                    for f in mat_files_in_dir)
            if has_resistance_data:
                measurement_dirs.append(root)
                logging.debug(f"  Added directory with MAT files: {root}")
    
    logging.debug(f"Total measurement directories found: {len(measurement_dirs)}")
    for md in measurement_dirs:
        logging.debug(f"  - {md}")
    
    data_files = []
    for measurement_dir in measurement_dirs:
        logging.debug(f"Searching for data files in: {measurement_dir}")
        
        # Look for MAT files with resistance ramp data using multiple patterns
        search_patterns = [
            file_pattern,  # Default pattern from config
            "*resistance*.mat",  # Any file with "resistance"
            "*ramp*.mat",  # Any file with "ramp"
            "*tc*.mat",  # Any file with "tc"
            "*.mat"  # All MAT files as fallback
        ]
        
        mat_files = []
        for pattern in search_patterns:
            pattern_files = glob.glob(os.path.join(measurement_dir, pattern))
            mat_files.extend(pattern_files)
        
        # Remove duplicates while preserving order
        mat_files = list(dict.fromkeys(mat_files))
        
        logging.debug(f"Found {len(mat_files)} MAT files in {measurement_dir}")
        
        for mat_file in mat_files:
            logging.debug(f"  Examining file: {os.path.basename(mat_file)}")
            
            # Quick check if file contains resistance data
            try:
                # Load MAT file and check for required data
                mat_data = loadmat(mat_file)
                
                # Extract the data from MAT file structure
                # Common keys in MATLAB files might be different
                data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                logging.debug(f"    MAT file keys: {data_keys}")
                
                # Try to find resistance and temperature data
                # This may need adjustment based on actual MAT file structure
                has_required_data = False
                for key in data_keys:
                    if isinstance(mat_data[key], np.ndarray) and mat_data[key].size > 10:
                        has_required_data = True
                        break
                
                if not has_required_data:
                    logging.debug(f"    Skipping - no suitable data arrays found")
                    continue
                
                logging.debug(f"    File has suitable data arrays")
                
            except Exception as e:
                logging.debug(f"    Error reading MAT file: {str(e)}")
                continue
            
            # Extract date from filename or directory
            try:
                # Try to extract date from filename
                date_match = re.search(date_pattern, mat_file)
                if date_match:
                    date_str = date_match.group(1)
                    measurement_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    logging.debug(f"    Extracted date from filename: {measurement_date}")
                else:
                    # Use file modification time as fallback
                    measurement_date = datetime.fromtimestamp(os.path.getmtime(mat_file))
                    logging.debug(f"    Using file modification time: {measurement_date}")
                
                data_files.append((mat_file, measurement_date))
                logging.debug(f"    Added data file: {os.path.basename(mat_file)}")
                
            except Exception as e:
                logging.warning(f"Could not parse date for {mat_file}: {str(e)}")
                # Still add the file with current time as fallback
                data_files.append((mat_file, datetime.now()))
    
    # Sort by measurement date (most recent first)
    data_files.sort(key=lambda x: x[1], reverse=True)
    
    logging.info(f"Found {len(data_files)} data files for device {device_name}")
    for file_path, date in data_files:
        logging.info(f"  - {os.path.basename(file_path)} ({date.strftime('%Y-%m-%d %H:%M')})")
    
    return data_files

def load_mat_file_as_dataframe(mat_file_path):
    """
    Load a MATLAB .mat file and convert it to a pandas DataFrame.
    
    Args:
        mat_file_path (str): Path to the .mat file
    
    Returns:
        pd.DataFrame: DataFrame with resistance and temperature data
    """
    try:
        # Load the MAT file
        mat_data = loadmat(mat_file_path)
        
        # Remove MATLAB metadata keys
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        logging.debug(f"MAT file keys: {data_keys}")
        
        # Try to identify resistance and temperature data
        # Look for specific key patterns first
        resistance_key = None
        temperature_key = None
        
        # Check for exact or close matches to expected keys
        for k in data_keys:
            k_lower = k.lower()
            if 'resistance' in k_lower and 'ohm' in k_lower:
                resistance_key = k
            elif 'temp' in k_lower and 'k' in k_lower and 'actual' in k_lower:
                temperature_key = k
            elif k_lower == 'resistanceohm':
                resistance_key = k
            elif k_lower == 'actualtempk':
                temperature_key = k
        
        # If we can't find specific keys, look for general patterns
        if not resistance_key or not temperature_key:
            resistance_keys = [k for k in data_keys if any(term in k.lower() for term in ['resistance', 'ohm', 'r'])]
            temperature_keys = [k for k in data_keys if any(term in k.lower() for term in ['temp', 'temperature', 'kelvin', 'k'])]
            
            if resistance_keys and not resistance_key:
                resistance_key = resistance_keys[0]
            if temperature_keys and not temperature_key:
                temperature_key = temperature_keys[0]
        
        # Fallback: assume first two numeric arrays are temperature and resistance
        if not resistance_key or not temperature_key:
            numeric_keys = []
            for k in data_keys:
                if isinstance(mat_data[k], np.ndarray) and mat_data[k].size > 10:
                    numeric_keys.append(k)
            
            if len(numeric_keys) >= 2:
                if not temperature_key:
                    temperature_key = numeric_keys[0]
                if not resistance_key:
                    resistance_key = numeric_keys[1]
        
        if not resistance_key or not temperature_key:
            raise ValueError(f"Could not identify resistance and temperature data in MAT file. Available keys: {data_keys}")
        
        logging.debug(f"Using resistance key: {resistance_key}")
        logging.debug(f"Using temperature key: {temperature_key}")
        
        # Extract the data arrays
        resistance_data = mat_data[resistance_key].flatten()
        temperature_data = mat_data[temperature_key].flatten()
        
        # Ensure both arrays have the same length
        min_length = min(len(resistance_data), len(temperature_data))
        resistance_data = resistance_data[:min_length]
        temperature_data = temperature_data[:min_length]
        
        # Create DataFrame with standard column names
        df = pd.DataFrame({
            'resistance_ohm': resistance_data,
            'actual_temp_K': temperature_data
        })
        
        logging.debug(f"Converted MAT file to DataFrame with {len(df)} data points")
        logging.debug(f"Resistance range: {df['resistance_ohm'].min():.2e} - {df['resistance_ohm'].max():.2e} Ohm")
        logging.debug(f"Temperature range: {df['actual_temp_K'].min():.2f} - {df['actual_temp_K'].max():.2f} K")
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading MAT file {mat_file_path}: {str(e)}")
        raise


def analyze_single_device(device_name, base_path):
    """
    Analyze all measurements for a single device.
    Returns dictionary with analysis results.
    """
    logging.info(f"Analyzing device {device_name}...")
    
    data_files = find_device_data_files(base_path, device_name)
    
    if not data_files:
        logging.warning(f"No data files found for device {device_name}")
        return {
            'device': device_name,
            'num_measurements': 0,
            'tc_values': [],
            'tc_mean': np.nan,
            'tc_std': np.nan,
            'best_tc': np.nan,
            'latest_date': None,
            'analysis_quality': 'no_data'
        }
    
    tc_values = []
    measurement_info = []
    
    for file_path, measurement_date in data_files:
        try:
            # Load the data - handle both CSV and MAT files
            if file_path.lower().endswith('.mat'):
                df = load_mat_file_as_dataframe(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Check if it has the required columns
            required_cols = CONFIG['file_patterns']['required_columns'] if CONFIG else ['resistance_ohm', 'actual_temp_K']
            if not all(col in df.columns for col in required_cols):
                logging.warning(f"Missing required columns in {file_path}")
                continue
            
            # Analyze critical temperature
            best_tc, tc_methods, tc_stats = analyze_critical_temperature_simple(df)
            
            if best_tc is not None:
                tc_values.append(best_tc)
                measurement_info.append({
                    'file': os.path.basename(file_path),
                    'date': measurement_date,
                    'tc': best_tc,
                    'method': tc_stats['best_method'] if tc_stats else 'unknown',
                    'quality': 'good' if tc_stats and tc_stats['std_tc'] < 0.05 else 'fair'
                })
                logging.info(f"  Found Tc = {best_tc:.3f} K from {os.path.basename(file_path)}")
            else:
                logging.warning(f"  Could not determine Tc from {os.path.basename(file_path)}")
                measurement_info.append({
                    'file': os.path.basename(file_path),
                    'date': measurement_date,
                    'tc': np.nan,
                    'method': 'failed',
                    'quality': 'poor'
                })
        
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {str(e)}")
            continue
    
    # Calculate statistics with configurable thresholds
    if tc_values:
        tc_mean = np.mean(tc_values)
        tc_std = np.std(tc_values) if len(tc_values) > 1 else 0
        best_tc = tc_values[0]  # Use most recent measurement
        
        # Determine analysis quality based on config thresholds
        good_threshold = CONFIG['quality']['good_std_threshold'] if CONFIG else 0.05
        fair_threshold = CONFIG['quality']['fair_std_threshold'] if CONFIG else 0.1
        
        if tc_std < good_threshold:
            analysis_quality = 'good'
        elif tc_std < fair_threshold:
            analysis_quality = 'fair'
        else:
            analysis_quality = 'poor'
    else:
        tc_mean = np.nan
        tc_std = np.nan
        best_tc = np.nan
        analysis_quality = 'no_valid_data'
    
    latest_date = data_files[0][1] if data_files else None
    
    result = {
        'device': device_name,
        'num_measurements': len(tc_values),
        'tc_values': tc_values,
        'tc_mean': tc_mean,
        'tc_std': tc_std,
        'best_tc': best_tc,
        'latest_date': latest_date,
        'analysis_quality': analysis_quality,
        'measurement_info': measurement_info
    }
    
    return result

def create_device_matrix(analysis_results):
    """
    Create a 7x7 matrix of Tc values for visualization.
    """
    # Initialize matrix with NaN
    tc_matrix = np.full((len(ROWS), len(COLS)), np.nan)
    quality_matrix = np.full((len(ROWS), len(COLS)), '', dtype='U20')
    count_matrix = np.zeros((len(ROWS), len(COLS)), dtype=int)
    
    for result in analysis_results:
        device_name = result['device']
        
        # Parse device name (e.g., "B5" -> row 1, col 4)
        if len(device_name) >= 2:
            row_char = device_name[0].upper()
            col_chars = device_name[1:]
            
            try:
                row_idx = ROWS.index(row_char)
                col_idx = COLS.index(col_chars)
                
                tc_matrix[row_idx, col_idx] = result['tc_mean']
                quality_matrix[row_idx, col_idx] = result['analysis_quality']
                count_matrix[row_idx, col_idx] = result['num_measurements']
                
            except ValueError:
                logging.warning(f"Could not parse device name: {device_name}")
    
    return tc_matrix, quality_matrix, count_matrix

def plot_tc_heatmap(tc_matrix, quality_matrix, count_matrix, output_dir):
    """
    Create a heatmap visualization of Tc values across the device matrix.
    """
    # Get plotting configuration
    if CONFIG:
        figsize = CONFIG['plotting']['figure_size']
        cmap_name = CONFIG['plotting']['heatmap_colormap']
        font_size = CONFIG['plotting']['font_size']
        dpi = CONFIG['output']['plot_dpi']
    else:
        figsize = [16, 12]
        cmap_name = 'RdYlBu_r'
        font_size = 10
        dpi = 300
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Tc values heatmap
    mask = np.isnan(tc_matrix)
    
    # Create custom colormap
    cmap = plt.cm.get_cmap(cmap_name)
    
    im1 = ax1.imshow(tc_matrix, cmap=cmap, aspect='equal', interpolation='nearest')
    
    # Add text annotations
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            if not mask[i, j]:
                text = f'{tc_matrix[i, j]:.2f}'
                ax1.text(j, i, text, ha='center', va='center', 
                        color='white' if tc_matrix[i, j] < np.nanmean(tc_matrix) else 'black',
                        fontsize=font_size, fontweight='bold')
            else:
                ax1.text(j, i, 'N/A', ha='center', va='center', 
                        color='gray', fontsize=font_size-2)
    
    ax1.set_xticks(range(len(COLS)))
    ax1.set_yticks(range(len(ROWS)))
    ax1.set_xticklabels(COLS)
    ax1.set_yticklabels(ROWS)
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.set_title('Critical Temperature (K) - Mean Values')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Tc (K)', rotation=270, labelpad=20)
    
    # Plot 2: Number of measurements
    im2 = ax2.imshow(count_matrix, cmap='Greens', aspect='equal')
    
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            text = f'{count_matrix[i, j]}'
            ax2.text(j, i, text, ha='center', va='center', 
                    color='white' if count_matrix[i, j] > np.max(count_matrix)/2 else 'black',
                    fontsize=font_size, fontweight='bold')
    
    ax2.set_xticks(range(len(COLS)))
    ax2.set_yticks(range(len(ROWS)))
    ax2.set_xticklabels(COLS)
    ax2.set_yticklabels(ROWS)
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.set_title('Number of Measurements per Device')
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Count', rotation=270, labelpad=20)
    
    # Plot 3: Tc distribution histogram
    valid_tc_values = tc_matrix[~np.isnan(tc_matrix)]
    if len(valid_tc_values) > 0:
        ax3.hist(valid_tc_values, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(valid_tc_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(valid_tc_values):.3f} K')
        ax3.axvline(np.median(valid_tc_values), color='orange', linestyle='--', 
                   label=f'Median: {np.median(valid_tc_values):.3f} K')
        ax3.set_xlabel('Critical Temperature (K)')
        ax3.set_ylabel('Number of Devices')
        ax3.set_title('Distribution of Critical Temperatures')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Quality assessment
    quality_colors = {'good': 0, 'fair': 1, 'poor': 2, 'no_data': 3, 'no_valid_data': 4}
    quality_numeric = np.full((len(ROWS), len(COLS)), 5)
    
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            quality = quality_matrix[i, j]
            if quality in quality_colors:
                quality_numeric[i, j] = quality_colors[quality]
    
    quality_cmap = plt.cm.RdYlGn_r
    im4 = ax4.imshow(quality_numeric, cmap=quality_cmap, aspect='equal', vmin=0, vmax=4)
    
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            quality = quality_matrix[i, j]
            ax4.text(j, i, quality.replace('_', '\n'), ha='center', va='center', 
                    fontsize=font_size-2, fontweight='bold')
    
    ax4.set_xticks(range(len(COLS)))
    ax4.set_yticks(range(len(ROWS)))
    ax4.set_xticklabels(COLS)
    ax4.set_yticklabels(ROWS)
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    ax4.set_title('Analysis Quality Assessment')
    
    plt.tight_layout()
    
    # Save the plot to plots subdirectory
    plots_dir = os.path.join(output_dir, "plots")
    plot_filename = os.path.join(plots_dir, 'device_matrix_tc_analysis.png')
    plt.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
    logging.info(f"Matrix heatmap saved: {plot_filename}")
    
    # Show plot if configured
    if CONFIG is None or CONFIG['plotting']['show_plots']:
        plt.show()
    else:
        plt.close()
    
    return plot_filename

def create_enhanced_matrix_plot(tc_matrix, quality_matrix, count_matrix, analysis_results, output_dir):
    """
    Create an enhanced matrix plot with better visualization and statistics.
    """
    # Get plotting configuration
    if CONFIG:
        figsize = [20, 14]
        cmap_name = CONFIG['plotting']['heatmap_colormap']
        font_size = CONFIG['plotting']['font_size']
        dpi = CONFIG['output']['plot_dpi']
    else:
        figsize = [20, 14]
        cmap_name = 'RdYlBu_r'
        font_size = 10
        dpi = 300
    
    fig = plt.figure(figsize=figsize)
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main Tc heatmap (large, top-left)
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    # Statistics plots
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_quality = fig.add_subplot(gs[1, 2])
    ax_stats = fig.add_subplot(gs[2, :])
    
    # Main heatmap
    mask = np.isnan(tc_matrix)
    cmap = plt.cm.get_cmap(cmap_name)
    
    im_main = ax_main.imshow(tc_matrix, cmap=cmap, aspect='equal', interpolation='nearest')
    
    # Add text annotations with device names and Tc values
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            device_name = f"{ROWS[i]}{COLS[j]}"
            if not mask[i, j]:
                text = f'{device_name}\n{tc_matrix[i, j]:.3f} K'
                ax_main.text(j, i, text, ha='center', va='center', 
                           color='white' if tc_matrix[i, j] < np.nanmean(tc_matrix) else 'black',
                           fontsize=font_size-1, fontweight='bold')
            else:
                ax_main.text(j, i, f'{device_name}\nN/A', ha='center', va='center', 
                           color='gray', fontsize=font_size-2)
    
    ax_main.set_xticks(range(len(COLS)))
    ax_main.set_yticks(range(len(ROWS)))
    ax_main.set_xticklabels(COLS)
    ax_main.set_yticklabels(ROWS)
    ax_main.set_xlabel('Column', fontsize=font_size+2)
    ax_main.set_ylabel('Row', fontsize=font_size+2)
    ax_main.set_title('Critical Temperature Matrix (K)', fontsize=font_size+4, fontweight='bold')
    
    # Add colorbar
    cbar_main = plt.colorbar(im_main, ax=ax_main, shrink=0.8)
    cbar_main.set_label('Tc (K)', rotation=270, labelpad=20, fontsize=font_size+2)
    
    # Histogram of Tc values
    valid_tc_values = tc_matrix[~np.isnan(tc_matrix)]
    if len(valid_tc_values) > 0:
        ax_hist.hist(valid_tc_values, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
        ax_hist.axvline(np.mean(valid_tc_values), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(valid_tc_values):.3f} K')
        ax_hist.axvline(np.median(valid_tc_values), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(valid_tc_values):.3f} K')
        ax_hist.set_xlabel('Tc (K)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Tc Distribution')
        ax_hist.legend(fontsize=font_size-2)
        ax_hist.grid(True, alpha=0.3)
    
    # Quality pie chart
    quality_counts = {}
    for result in analysis_results:
        quality = result['analysis_quality']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    if quality_counts:
        colors = ['green', 'yellow', 'orange', 'red', 'gray']
        quality_labels = ['Good', 'Fair', 'Poor', 'No Data', 'No Valid Data']
        quality_keys = ['good', 'fair', 'poor', 'no_data', 'no_valid_data']
        
        wedges, texts, autotexts = ax_quality.pie(
            [quality_counts.get(key, 0) for key in quality_keys],
            labels=[f'{label}\n({quality_counts.get(key, 0)})' for label, key in zip(quality_labels, quality_keys)],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax_quality.set_title('Analysis Quality')
    
    # Statistics table
    ax_stats.axis('off')
    
    if len(valid_tc_values) > 0:
        stats_data = [
            ['Total Devices', f'{len(ROWS) * len(COLS)}'],
            ['Devices with Data', f'{len([r for r in analysis_results if r["num_measurements"] > 0])}'],
            ['Devices with Valid Tc', f'{len(valid_tc_values)}'],
            ['Success Rate', f'{len(valid_tc_values)/(len(ROWS) * len(COLS))*100:.1f}%'],
            ['Mean Tc', f'{np.mean(valid_tc_values):.3f} +/- {np.std(valid_tc_values):.3f} K'],
            ['Tc Range', f'{np.min(valid_tc_values):.3f} - {np.max(valid_tc_values):.3f} K'],
            ['Coefficient of Variation', f'{np.std(valid_tc_values)/np.mean(valid_tc_values)*100:.1f}%'],
        ]
    else:
        stats_data = [['No valid data found', '']]
    
    table = ax_stats.table(cellText=stats_data, 
                          colLabels=['Statistic', 'Value'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(font_size+1)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    # Add title
    fig.suptitle(f'Device Matrix Tc Analysis - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                fontsize=font_size+6, fontweight='bold')
    
    # Save the enhanced plot to plots subdirectory
    plots_dir = os.path.join(output_dir, "plots")
    plot_filename = os.path.join(plots_dir, 'enhanced_matrix_analysis.png')
    plt.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
    logging.info(f"Enhanced matrix plot saved: {plot_filename}")
    
    # Show plot if configured
    if CONFIG is None or CONFIG['plotting']['show_plots']:
        plt.show()
    else:
        plt.close()
    
    return plot_filename

def create_individual_device_plots(analysis_results, output_dir):
    """
    Create individual plots for devices with multiple measurements.
    """
    devices_with_multiple = [r for r in analysis_results if r['num_measurements'] > 1]
    
    if not devices_with_multiple:
        logging.info("No devices with multiple measurements found")
        return
    
    # Create subplot grid
    n_devices = len(devices_with_multiple)
    cols = min(4, n_devices)
    rows = (n_devices + cols - 1) // cols
    
    # Handle the single device case properly
    if n_devices == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        axes = [ax]  # Make it a list for consistent indexing
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        
        # Ensure axes is always a flat array for consistent indexing
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
    
    for idx, result in enumerate(devices_with_multiple):
        ax = axes[idx]
        
        device_name = result['device']
        tc_values = result['tc_values']
        dates = [info['date'] for info in result['measurement_info'] if not np.isnan(info['tc'])]
        
        if tc_values and dates:
            # Sort by date
            sorted_data = sorted(zip(dates, tc_values))
            dates_sorted, tc_sorted = zip(*sorted_data)
            
            ax.plot(dates_sorted, tc_sorted, 'bo-', markersize=6, linewidth=2)
            ax.set_title(f'Device {device_name}\n{len(tc_values)} measurements')
            ax.set_ylabel('Tc (K)')
            ax.grid(True, alpha=0.3)
            
            # Add mean line
            ax.axhline(y=result['tc_mean'], color='red', linestyle='--', 
                      label=f'Mean: {result["tc_mean"]:.3f} K')
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
            
            if len(tc_values) > 1:
                ax.legend()
    
    # Hide empty subplots
    for idx in range(n_devices, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    plots_dir = os.path.join(output_dir, "plots")
    plot_filename = os.path.join(plots_dir, 'individual_device_trends.png')
    dpi = CONFIG['output']['plot_dpi'] if CONFIG else 300
    plt.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
    logging.info(f"Individual device trends plot saved: {plot_filename}")
    
    # Show plot if configured
    if CONFIG is None or CONFIG['plotting']['show_plots']:
        plt.show()
    else:
        plt.close()
    
    return plot_filename

def save_summary_csv(analysis_results, tc_matrix, output_dir):
    """
    Save comprehensive summary to CSV files in data subdirectory.
    """
    data_dir = os.path.join(output_dir, "data")
    
    # Summary CSV with one row per device
    summary_data = []
    for result in analysis_results:
        # Handle latest_date safely
        latest_date = result.get('latest_date')
        if latest_date and hasattr(latest_date, 'strftime'):
            latest_date_str = latest_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            latest_date_str = 'N/A'
            
        summary_data.append({
            'Device': result['device'],
            'Num_Measurements': result.get('num_measurements', 0),
            'Tc_Mean_K': result.get('tc_mean', np.nan),
            'Tc_Std_K': result.get('tc_std', np.nan),
            'Best_Tc_K': result.get('best_tc', np.nan),
            'Latest_Date': latest_date_str,
            'Analysis_Quality': result.get('analysis_quality', 'unknown')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = os.path.join(data_dir, 'device_matrix_summary.csv')
    summary_df.to_csv(summary_filename, index=False)
    logging.info(f"Summary CSV saved: {summary_filename}")
    
    # Detailed CSV with one row per measurement
    detailed_data = []
    for result in analysis_results:
        device_name = result['device']
        # Check if measurement_info exists and has data
        measurement_info = result.get('measurement_info', [])
        if measurement_info:
            for info in measurement_info:
                # Handle both datetime objects and string dates
                date_str = info['date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(info['date'], 'strftime') else str(info['date'])
                detailed_data.append({
                    'Device': device_name,
                    'File': info['file'],
                    'Date': date_str,
                    'Tc_K': info['tc'],
                    'Method': info['method'],
                    'Quality': info['quality']
                })
        else:
            # Add a placeholder entry for devices with no measurements
            detailed_data.append({
                'Device': device_name,
                'File': 'No data found',
                'Date': 'N/A',
                'Tc_K': np.nan,
                'Method': 'N/A',
                'Quality': 'no_data'
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_filename = os.path.join(data_dir, 'device_matrix_detailed.csv')
    detailed_df.to_csv(detailed_filename, index=False)
    logging.info(f"Detailed CSV saved: {detailed_filename}")
    
    # Matrix CSV (for easy import into other tools)
    matrix_df = pd.DataFrame(tc_matrix, index=ROWS, columns=COLS)
    matrix_filename = os.path.join(data_dir, 'tc_matrix.csv')
    matrix_df.to_csv(matrix_filename)
    logging.info(f"Matrix CSV saved: {matrix_filename}")
    
    return summary_filename, detailed_filename, matrix_filename

def print_analysis_summary(analysis_results, tc_matrix):
    """
    Print a comprehensive analysis summary to the log.
    """
    logging.info("=" * 60)
    logging.info("DEVICE MATRIX TC ANALYSIS SUMMARY")
    logging.info("=" * 60)
    
    # Overall statistics
    total_devices = len(ROWS) * len(COLS)
    analyzed_devices = len([r for r in analysis_results if r['num_measurements'] > 0])
    devices_with_tc = len([r for r in analysis_results if not np.isnan(r['tc_mean'])])
    
    logging.info(f"Total devices in matrix: {total_devices}")
    logging.info(f"Devices with data: {analyzed_devices}")
    logging.info(f"Devices with valid Tc: {devices_with_tc}")
    logging.info(f"Success rate: {devices_with_tc/total_devices*100:.1f}%")
    
    # Tc statistics
    valid_tc_values = tc_matrix[~np.isnan(tc_matrix)]
    if len(valid_tc_values) > 0:
        logging.info(f"\nTc Statistics:")
        logging.info(f"  Mean: {np.mean(valid_tc_values):.3f} K")
        logging.info(f"  Median: {np.median(valid_tc_values):.3f} K")
        logging.info(f"  Std Dev: {np.std(valid_tc_values):.3f} K")
        logging.info(f"  Min: {np.min(valid_tc_values):.3f} K")
        logging.info(f"  Max: {np.max(valid_tc_values):.3f} K")
        logging.info(f"  Range: {np.max(valid_tc_values) - np.min(valid_tc_values):.3f} K")
    
    # Quality assessment
    quality_counts = {}
    for result in analysis_results:
        quality = result['analysis_quality']
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    logging.info(f"\nQuality Assessment:")
    for quality, count in quality_counts.items():
        logging.info(f"  {quality}: {count} devices")
    
    # Devices with multiple measurements
    multiple_measurements = [r for r in analysis_results if r['num_measurements'] > 1]
    logging.info(f"\nDevices with multiple measurements: {len(multiple_measurements)}")
    for result in multiple_measurements:
        tc_values = result['tc_values']
        logging.info(f"  {result['device']}: {len(tc_values)} measurements, "
                    f"Tc = {result['tc_mean']:.3f} +/- {result['tc_std']:.3f} K")
    
    logging.info("=" * 60)

def create_analysis_readme(analysis_results, tc_matrix, output_dir):
    """
    Create a comprehensive README file explaining the analysis and results.
    """
    readme_filename = os.path.join(output_dir, 'README.md')
    
    with open(readme_filename, 'w') as f:
        f.write("# Device Matrix Tc Analysis Results\n\n")
        f.write(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration information
        f.write("## Configuration\n\n")
        if CONFIG:
            f.write(f"- **Base Data Path**: {CONFIG['data_source']['base_path']}\n")
            f.write(f"- **Sample**: {CONFIG['data_source']['sample_name']}\n")
            f.write(f"- **Device Type**: {CONFIG['data_source']['device_type']}\n")
            f.write(f"- **Measurement Type**: {CONFIG['data_source']['measurement_type']}\n")
            f.write(f"- **Matrix Size**: {len(ROWS)}x{len(COLS)} = {len(ROWS)*len(COLS)} devices\n")
        else:
            f.write("- Configuration file not found, using defaults\n")
        
        f.write("\n## Analysis Summary\n\n")
        
        # Overall statistics
        total_devices = len(ROWS) * len(COLS)
        analyzed_devices = len([r for r in analysis_results if r['num_measurements'] > 0])
        devices_with_tc = len([r for r in analysis_results if not np.isnan(r['tc_mean'])])
        
        f.write(f"- **Total devices in matrix**: {total_devices}\n")
        f.write(f"- **Devices with measurement data**: {analyzed_devices}\n")
        f.write(f"- **Devices with valid Tc measurements**: {devices_with_tc}\n")
        f.write(f"- **Overall success rate**: {devices_with_tc/total_devices*100:.1f}%\n\n")
        
        # Tc statistics
        valid_tc_values = tc_matrix[~np.isnan(tc_matrix)]
        if len(valid_tc_values) > 0:
            f.write("### Critical Temperature Statistics\n\n")
            f.write(f"- **Mean Tc**: {np.mean(valid_tc_values):.3f} +/- {np.std(valid_tc_values):.3f} K\n")
            f.write(f"- **Median Tc**: {np.median(valid_tc_values):.3f} K\n")
            f.write(f"- **Tc Range**: {np.min(valid_tc_values):.3f} - {np.max(valid_tc_values):.3f} K\n")
            f.write(f"- **Coefficient of Variation**: {np.std(valid_tc_values)/np.mean(valid_tc_values)*100:.1f}%\n\n")
        
        # Quality assessment
        f.write("### Quality Assessment\n\n")
        quality_counts = {}
        for result in analysis_results:
            quality = result['analysis_quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        for quality, count in quality_counts.items():
            f.write(f"- **{quality.replace('_', ' ').title()}**: {count} devices\n")
        
        f.write("\n## Files Generated\n\n")
        f.write("### Plots (in `plots/` directory)\n")
        f.write("- `device_matrix_tc_analysis.png` - Main heatmap visualization\n")
        f.write("- `enhanced_matrix_analysis.png` - Enhanced visualization with statistics\n")
        f.write("- `individual_device_trends.png` - Trends for devices with multiple measurements\n\n")
        f.write("### Data Files (in `data/` directory)\n")
        f.write("- `device_matrix_summary.csv` - Summary data (one row per device)\n")
        f.write("- `device_matrix_detailed.csv` - Detailed data (one row per measurement)\n")
        f.write("- `tc_matrix.csv` - Matrix format for external analysis\n")
        f.write("- `device_matrix_analysis_YYYYMMDD_HHMMSS.log` - Analysis log file\n\n")
        f.write("### Other Files\n")
        f.write("- This README file\n\n")
        
        # Methodology
        f.write("## Analysis Methodology\n\n")
        f.write("### Critical Temperature Determination\n\n")
        f.write("Three methods are used to determine Tc from resistance vs temperature data:\n\n")
        f.write("1. **Derivative Method**: Finds the temperature where dR/dT is maximum\n")
        f.write("2. **Threshold Method**: Finds where resistance exceeds 10% of the transition\n")
        f.write("3. **Midpoint Method**: Finds the 50% point of the resistance transition\n\n")
        f.write("The best estimate is chosen based on consistency between methods.\n\n")
        
        f.write("### Quality Criteria\n\n")
        if CONFIG:
            f.write(f"- **Good**: Standard deviation < {CONFIG['quality']['good_std_threshold']} K\n")
            f.write(f"- **Fair**: Standard deviation < {CONFIG['quality']['fair_std_threshold']} K\n")
        else:
            f.write("- **Good**: Standard deviation < 0.05 K\n")
            f.write("- **Fair**: Standard deviation < 0.1 K\n")
        f.write("- **Poor**: Higher standard deviation or analysis issues\n")
        f.write("- **No Data**: No measurement files found\n")
        f.write("- **No Valid Data**: Files found but Tc could not be determined\n\n")
        
        # Device-specific results
        f.write("## Device-Specific Results\n\n")
        f.write("| Device | Num Measurements | Tc Mean (K) | Tc Std (K) | Quality | Latest Date |\n")
        f.write("|--------|------------------|-------------|-------------|---------|-------------|\n")
        
        for result in sorted(analysis_results, key=lambda x: x['device']):
            device = result['device']
            num_meas = result['num_measurements']
            tc_mean = f"{result['tc_mean']:.3f}" if not np.isnan(result['tc_mean']) else "N/A"
            tc_std = f"{result['tc_std']:.3f}" if not np.isnan(result['tc_std']) else "N/A"
            quality = result['analysis_quality']
            latest_date = result['latest_date'].strftime('%Y-%m-%d') if result['latest_date'] else "N/A"
            
            f.write(f"| {device} | {num_meas} | {tc_mean} | {tc_std} | {quality} | {latest_date} |\n")
        
        f.write("\n## Notes\n\n")
        f.write("- For devices with multiple measurements, the mean Tc is reported\n")
        f.write("- The 'Best Tc' uses the most recent measurement for each device\n")
        f.write("- Data smoothing is applied to reduce noise before analysis\n")
        f.write("- Matrix positions correspond to physical device locations\n")
    
    logging.info(f"README file created: {readme_filename}")
    return readme_filename

def debug_directory_structure(base_path, max_devices_to_check=5):
    """
    Debug function to examine the directory structure and help identify data locations.
    """
    logging.info("DEBUGGING DIRECTORY STRUCTURE")
    logging.info("=" * 50)
    logging.info(f"Base path: {base_path}")
    
    if not os.path.exists(base_path):
        logging.error(f"Base path does not exist: {base_path}")
        return
    
    # List all directories in base path
    try:
        base_contents = os.listdir(base_path)
        device_dirs = [d for d in base_contents if os.path.isdir(os.path.join(base_path, d))]
        logging.info(f"Found {len(device_dirs)} directories in base path:")
        for d in sorted(device_dirs):
            logging.info(f"  - {d}")
        
        # Check a few device directories in detail
        sample_devices = device_dirs[:max_devices_to_check]
        logging.info(f"\nExamining structure of first {len(sample_devices)} device directories:")
        
        for device_dir in sample_devices:
            device_path = os.path.join(base_path, device_dir)
            logging.info(f"\n--- Device: {device_dir} ---")
            logging.info(f"Path: {device_path}")
            
            try:
                # Walk through the directory structure
                for root, dirs, files in os.walk(device_path):
                    level = root.replace(device_path, '').count(os.sep)
                    indent = '  ' * level
                    logging.info(f"{indent}{os.path.basename(root)}/")
                    
                    # Show subdirectories
                    sub_indent = '  ' * (level + 1)
                    for d in dirs:
                        logging.info(f"{sub_indent}{d}/")
                    
                    # Show CSV files
                    csv_files = [f for f in files if f.lower().endswith('.csv')]
                    if csv_files:
                        logging.info(f"{sub_indent}CSV files:")
                        for f in csv_files[:3]:  # Show first 3 CSV files
                            logging.info(f"{sub_indent}  - {f}")
                        if len(csv_files) > 3:
                            logging.info(f"{sub_indent}  ... and {len(csv_files)-3} more")
                    
                    # Show MAT files
                    mat_files = [f for f in files if f.lower().endswith('.mat')]
                    if mat_files:
                        logging.info(f"{sub_indent}MAT files:")
                        for f in mat_files[:3]:  # Show first 3 MAT files
                            logging.info(f"{sub_indent}  - {f}")
                        if len(mat_files) > 3:
                            logging.info(f"{sub_indent}  ... and {len(mat_files)-3} more")
                    
                    # Don't go too deep
                    if level > 3:
                        break
                        
            except Exception as e:
                logging.error(f"Error examining {device_dir}: {str(e)}")
    
    except Exception as e:
        logging.error(f"Error listing base directory: {str(e)}")
    
    logging.info("=" * 50)

def check_specific_device(base_path, device_name):
    """
    Detailed check for a specific device to understand its data structure.
    """
    logging.info(f"DETAILED CHECK FOR DEVICE: {device_name}")
    logging.info("=" * 50)
    
    device_path = os.path.join(base_path, device_name)
    logging.info(f"Device path: {device_path}")
    
    if not os.path.exists(device_path):
        logging.warning(f"Device directory does not exist: {device_path}")
        
        # Check if there are similar named directories
        try:
            base_contents = os.listdir(base_path)
            similar = [d for d in base_contents if device_name.lower() in d.lower() or d.lower() in device_name.lower()]
            if similar:
                logging.info(f"Found similar directory names: {similar}")
        except:
            pass
        return
    
    logging.info(f"Device directory exists!")
    
    # Examine all files and subdirectories
    all_csv_files = []
    all_mat_files = []
    all_dirs = []
    
    for root, dirs, files in os.walk(device_path):
        rel_path = os.path.relpath(root, device_path)
        level = rel_path.count(os.sep) if rel_path != '.' else 0
        
        logging.info(f"{'  ' * level}Directory: {os.path.basename(root) if level > 0 else '(root)'}")
        
        # Track directories
        for d in dirs:
            full_dir = os.path.join(root, d)
            all_dirs.append(full_dir)
            logging.info(f"{'  ' * (level+1)}+ {d}/")
        
        # Track CSV files
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        for f in csv_files:
            full_file = os.path.join(root, f)
            all_csv_files.append(full_file)
            file_size = os.path.getsize(full_file)
            file_date = datetime.fromtimestamp(os.path.getmtime(full_file))
            logging.info(f"{'  ' * (level+1)}- {f} ({file_size} bytes, {file_date.strftime('%Y-%m-%d %H:%M')})")
        
        # Track MAT files
        mat_files = [f for f in files if f.lower().endswith('.mat')]
        for f in mat_files:
            full_file = os.path.join(root, f)
            all_mat_files.append(full_file)
            file_size = os.path.getsize(full_file)
            file_date = datetime.fromtimestamp(os.path.getmtime(full_file))
            logging.info(f"{'  ' * (level+1)}- {f} ({file_size} bytes, {file_date.strftime('%Y-%m-%d %H:%M')})")
    
    logging.info(f"\nSUMMARY FOR {device_name}:")
    logging.info(f"Total subdirectories: {len(all_dirs)}")
    logging.info(f"Total CSV files: {len(all_csv_files)}")
    logging.info(f"Total MAT files: {len(all_mat_files)}")
    
    if all_csv_files:
        logging.info(f"CSV files found:")
        for csv_file in all_csv_files:
            # Try to check if it contains resistance data
            try:
                df_sample = pd.read_csv(csv_file, nrows=3)
                cols = list(df_sample.columns)
                has_resistance = any('resistance' in col.lower() for col in cols)
                has_temp = any('temp' in col.lower() for col in cols)
                logging.info(f"  - {os.path.basename(csv_file)}: columns={cols[:5]}... resistance={has_resistance}, temp={has_temp}")
            except Exception as e:
                logging.info(f"  - {os.path.basename(csv_file)}: Error reading file - {str(e)}")
    
    if all_mat_files:
        logging.info(f"MAT files found:")
        for mat_file in all_mat_files:
            # Try to check MAT file contents
            try:
                mat_data = loadmat(mat_file)
                data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                has_resistance = any('resistance' in k.lower() or 'r' in k.lower() for k in data_keys)
                has_temp = any('temp' in k.lower() or 't' in k.lower() for k in data_keys)
                logging.info(f"  - {os.path.basename(mat_file)}: keys={data_keys[:5]}... resistance={has_resistance}, temp={has_temp}")
            except Exception as e:
                logging.info(f"  - {os.path.basename(mat_file)}: Error reading file - {str(e)}")
    
    logging.info("=" * 50)

# Convenience functions for interactive use
def run_debug_mode(base_path=None):
    """Run debug mode to examine directory structure."""
    if base_path is None:
        base_path = BASE_DATA_PATH
    print(f"Running debug mode on base path: {base_path}")
    debug_directory_structure(base_path)

def check_device(device_name, base_path=None):
    """Check a specific device for debugging."""
    if base_path is None:
        base_path = BASE_DATA_PATH
    print(f"Checking specific device: {device_name}")
    check_specific_device(base_path, device_name)

def test_single_device(device_name, base_path=None):
    """Test analysis on a single device."""
    if base_path is None:
        base_path = BASE_DATA_PATH
    print(f"Testing analysis on device: {device_name}")
    result = analyze_single_device(device_name, base_path)
    print(f"Result: {result}")
    return result

def main():
    """
    Main function to analyze the entire device matrix.
    """
    logging.info("Starting Device Matrix Tc Analysis")
    logging.info(f"Base data path: {BASE_DATA_PATH}")
    logging.info(f"Looking for measurement type: {MEASUREMENT_TYPE}")
    
    # Check if base path exists
    if not os.path.exists(BASE_DATA_PATH):
        logging.error(f"Base data path does not exist: {BASE_DATA_PATH}")
        return
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Generate all device names (A1 to G7)
    all_devices = [f"{row}{col}" for row in ROWS for col in COLS]
    logging.info(f"Analyzing {len(all_devices)} devices: {all_devices[0]} to {all_devices[-1]}")
    
    # Analyze each device
    analysis_results = []
    for device_name in all_devices:
        result = analyze_single_device(device_name, BASE_DATA_PATH)
        analysis_results.append(result)
    
    # Create device matrix
    tc_matrix, quality_matrix, count_matrix = create_device_matrix(analysis_results)
    
    # Create visualizations
    logging.info("Creating visualizations...")
    plot_tc_heatmap(tc_matrix, quality_matrix, count_matrix, output_dir)
    create_enhanced_matrix_plot(tc_matrix, quality_matrix, count_matrix, analysis_results, output_dir)
    create_individual_device_plots(analysis_results, output_dir)
    
    # Save CSV files
    logging.info("Saving CSV files...")
    save_summary_csv(analysis_results, tc_matrix, output_dir)
    
    # Create README
    logging.info("Creating README...")
    create_analysis_readme(analysis_results, tc_matrix, output_dir)
    
    # Print summary
    print_analysis_summary(analysis_results, tc_matrix)
    
    logging.info(f"Analysis complete! Results saved to: {output_dir}")
    
    return analysis_results, tc_matrix, output_dir

if __name__ == "__main__":
    try:
        results, matrix, output_dir = main()
        logging.info("Device matrix analysis completed successfully!")
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"Found Tc values for {len([r for r in results if not np.isnan(r['tc_mean'])])} devices")
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

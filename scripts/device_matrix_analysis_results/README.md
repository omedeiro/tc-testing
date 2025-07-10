# Device Matrix Tc Analysis Results

Analysis performed on: 2025-07-10 13:20:41

## Configuration

- **Base Data Path**: S:\SC\Measurements\SPG806\C6
- **Sample**: SPG806
- **Device Type**: C6
- **Measurement Type**: resistance_ramp
- **Matrix Size**: 7×7 = 49 devices

## Analysis Summary

- **Total devices in matrix**: 49
- **Devices with measurement data**: 1
- **Devices with valid Tc measurements**: 1
- **Overall success rate**: 2.0%

### Critical Temperature Statistics

- **Mean Tc**: 8.631 ± 0.000 K
- **Median Tc**: 8.631 K
- **Tc Range**: 8.631 - 8.631 K
- **Coefficient of Variation**: 0.0%

### Quality Assessment

- **No Data**: 48 devices
- **Good**: 1 devices

## Files Generated

### Plots (in `plots/` directory)
- `device_matrix_tc_analysis.png` - Main heatmap visualization
- `enhanced_matrix_analysis.png` - Enhanced visualization with statistics
- `individual_device_trends.png` - Trends for devices with multiple measurements

### Data Files (in `data/` directory)
- `device_matrix_summary.csv` - Summary data (one row per device)
- `device_matrix_detailed.csv` - Detailed data (one row per measurement)
- `tc_matrix.csv` - Matrix format for external analysis

### Other Files
- Analysis log file
- This README file

## Analysis Methodology

### Critical Temperature Determination

Three methods are used to determine Tc from resistance vs temperature data:

1. **Derivative Method**: Finds the temperature where dR/dT is maximum
2. **Threshold Method**: Finds where resistance exceeds 10% of the transition
3. **Midpoint Method**: Finds the 50% point of the resistance transition

The best estimate is chosen based on consistency between methods.

### Quality Criteria

- **Good**: Standard deviation < 0.05 K
- **Fair**: Standard deviation < 0.1 K
- **Poor**: Higher standard deviation or analysis issues
- **No Data**: No measurement files found
- **No Valid Data**: Files found but Tc could not be determined

## Device-Specific Results

| Device | Num Measurements | Tc Mean (K) | Tc Std (K) | Quality | Latest Date |
|--------|------------------|-------------|-------------|---------|-------------|
| A1 | 0 | N/A | N/A | no_data | N/A |
| A2 | 0 | N/A | N/A | no_data | N/A |
| A3 | 0 | N/A | N/A | no_data | N/A |
| A4 | 0 | N/A | N/A | no_data | N/A |
| A5 | 0 | N/A | N/A | no_data | N/A |
| A6 | 0 | N/A | N/A | no_data | N/A |
| A7 | 0 | N/A | N/A | no_data | N/A |
| B1 | 0 | N/A | N/A | no_data | N/A |
| B2 | 0 | N/A | N/A | no_data | N/A |
| B3 | 0 | N/A | N/A | no_data | N/A |
| B4 | 0 | N/A | N/A | no_data | N/A |
| B5 | 0 | N/A | N/A | no_data | N/A |
| B6 | 1 | 8.631 | 0.000 | good | 2025-07-10 |
| B7 | 0 | N/A | N/A | no_data | N/A |
| C1 | 0 | N/A | N/A | no_data | N/A |
| C2 | 0 | N/A | N/A | no_data | N/A |
| C3 | 0 | N/A | N/A | no_data | N/A |
| C4 | 0 | N/A | N/A | no_data | N/A |
| C5 | 0 | N/A | N/A | no_data | N/A |
| C6 | 0 | N/A | N/A | no_data | N/A |
| C7 | 0 | N/A | N/A | no_data | N/A |
| D1 | 0 | N/A | N/A | no_data | N/A |
| D2 | 0 | N/A | N/A | no_data | N/A |
| D3 | 0 | N/A | N/A | no_data | N/A |
| D4 | 0 | N/A | N/A | no_data | N/A |
| D5 | 0 | N/A | N/A | no_data | N/A |
| D6 | 0 | N/A | N/A | no_data | N/A |
| D7 | 0 | N/A | N/A | no_data | N/A |
| E1 | 0 | N/A | N/A | no_data | N/A |
| E2 | 0 | N/A | N/A | no_data | N/A |
| E3 | 0 | N/A | N/A | no_data | N/A |
| E4 | 0 | N/A | N/A | no_data | N/A |
| E5 | 0 | N/A | N/A | no_data | N/A |
| E6 | 0 | N/A | N/A | no_data | N/A |
| E7 | 0 | N/A | N/A | no_data | N/A |
| F1 | 0 | N/A | N/A | no_data | N/A |
| F2 | 0 | N/A | N/A | no_data | N/A |
| F3 | 0 | N/A | N/A | no_data | N/A |
| F4 | 0 | N/A | N/A | no_data | N/A |
| F5 | 0 | N/A | N/A | no_data | N/A |
| F6 | 0 | N/A | N/A | no_data | N/A |
| F7 | 0 | N/A | N/A | no_data | N/A |
| G1 | 0 | N/A | N/A | no_data | N/A |
| G2 | 0 | N/A | N/A | no_data | N/A |
| G3 | 0 | N/A | N/A | no_data | N/A |
| G4 | 0 | N/A | N/A | no_data | N/A |
| G5 | 0 | N/A | N/A | no_data | N/A |
| G6 | 0 | N/A | N/A | no_data | N/A |
| G7 | 0 | N/A | N/A | no_data | N/A |

## Notes

- For devices with multiple measurements, the mean Tc is reported
- The 'Best Tc' uses the most recent measurement for each device
- Data smoothing is applied to reduce noise before analysis
- Matrix positions correspond to physical device locations

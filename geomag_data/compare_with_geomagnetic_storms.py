import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import requests
import json
import matplotlib.dates as mdates
import pytz

# Define Kp thresholds for different storm levels
G_STORM_LEVELS = {
    'G1': 5,   # Kp = 5: Minor storm
    'G2': 6,   # Kp = 6: Moderate storm
    'G3': 7,   # Kp = 7: Strong storm
    'G4': 8,   # Kp = 8: Severe storm
    'G5': 9    # Kp = 9: Extreme storm
}

# Keep some hardcoded storms for periods we might not get NOAA data (historical or future)
HISTORICAL_STORMS = [
    ('2025-03-21', '2025-03-22', 'G2 Storm'),
    ('2025-04-10', '2025-04-12', 'G3 Storm'),
    ('2025-05-05', '2025-05-07', 'G4 Severe Storm')
]

def load_processed_data(file_path='analysis_results/processed_data.csv'):
    """Load the previously processed magnetometer data"""
    print(f"Loading processed data from {file_path}...")
    try:
        # Use flexible datetime parser to handle various formats
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = df.set_index('timestamp')
        
        # Print information about the loaded data
        print(f"Successfully loaded {len(df)} records spanning from {df.index.min()} to {df.index.max()}")
        print(f"Available columns: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

def fetch_noaa_data():
    """Fetch the most recent geomagnetic data from NOAA Space Weather API"""
    print("Fetching NOAA geomagnetic data...")
    try:
        # NOAA Planetary K-index API endpoint
        kp_api_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        
        # Historical NOAA Kp index data (past 30 days)
        historical_kp_url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
        
        # First try to get the historical data (past 30 days, more complete)
        response = requests.get(historical_kp_url)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Successfully fetched historical NOAA data: {len(data)} records")
        else:
            # Fallback to real-time data
            response = requests.get(kp_api_url)
            if response.status_code == 200:
                data = response.json()
                # Skip header row
                data = data[1:]
                print(f"Successfully fetched real-time NOAA data: {len(data)} records")
            else:
                print(f"Failed to fetch NOAA data: HTTP {response.status_code}")
                return None
        
        # Process the API response into a dataframe
        kp_data = []
        
        # Use UTC timezone for consistency
        utc = pytz.UTC
        
        # The API format is slightly different between endpoints
        for item in data:
            # Handle different formats between APIs
            if isinstance(item, dict):
                time_tag = item.get('time_tag')
                kp_index = item.get('kp_index')
            else:
                # For the array format
                time_tag = item[0]
                kp_index = item[1]
            
            if time_tag and kp_index is not None:
                try:
                    # Convert to datetime and numeric Kp
                    time_tag = pd.to_datetime(time_tag)
                    # Make timezone aware (UTC)
                    if time_tag.tzinfo is None:
                        time_tag = time_tag.replace(tzinfo=utc)
                    kp_index = float(kp_index)
                    
                    kp_data.append({
                        'timestamp': time_tag,
                        'kp_index': kp_index
                    })
                except Exception as e:
                    print(f"Error parsing NOAA data: {e}")
                    continue
        
        if kp_data:
            kp_df = pd.DataFrame(kp_data)
            kp_df = kp_df.set_index('timestamp')
            kp_df = kp_df.sort_index()
            print(f"Processed NOAA data: {len(kp_df)} valid records")
            return kp_df
        else:
            print("No valid NOAA data available")
            return None
    except Exception as e:
        print(f"Error fetching NOAA data: {e}")
        return None

def identify_storm_periods(kp_data):
    """Identify storm periods from Kp index data"""
    if kp_data is None or kp_data.empty:
        print("No Kp data available to identify storms")
        return []
    
    print("Identifying storm periods from Kp data...")
    
    # Storm periods to return
    storm_periods = []
    
    # Find periods where Kp >= 5 (G1 or higher)
    storm_threshold = G_STORM_LEVELS['G1']
    storm_mask = kp_data['kp_index'] >= storm_threshold
    
    if not storm_mask.any():
        print("No geomagnetic storms found in the NOAA data")
        return []
    
    # Get the indices where storms begin and end
    storm_starts = np.where(np.diff(np.concatenate(([False], storm_mask.values))) == 1)[0]
    storm_ends = np.where(np.diff(np.concatenate((storm_mask.values, [False]))) == -1)[0]
    
    # Safety check: ensure we have the same number of starts and ends
    if len(storm_starts) != len(storm_ends):
        print("Warning: Unequal number of storm starts and ends, adjusting...")
        min_len = min(len(storm_starts), len(storm_ends))
        storm_starts = storm_starts[:min_len]
        storm_ends = storm_ends[:min_len]
    
    # Process each storm period
    for i in range(len(storm_starts)):
        start_idx = storm_starts[i]
        end_idx = storm_ends[i]
        
        # Get the timestamps
        start_time = kp_data.index[start_idx]
        end_time = kp_data.index[end_idx]
        
        # Find the maximum Kp index in this period to determine storm level
        max_kp = kp_data['kp_index'].iloc[start_idx:end_idx+1].max()
        
        # Determine storm level
        storm_level = "G1"  # Default: Minor storm
        for level, threshold in sorted(G_STORM_LEVELS.items(), key=lambda x: x[1], reverse=True):
            if max_kp >= threshold:
                storm_level = level
                break
        
        # Create a description
        description = f"{storm_level} Storm (Kp Max: {max_kp:.1f})"
        
        # Add to the list
        storm_periods.append((start_time, end_time, description))
    
    print(f"Identified {len(storm_periods)} storm periods from NOAA data")
    for start, end, desc in storm_periods:
        print(f"  * {desc}: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
    
    return storm_periods

def plot_data_with_storm_periods(mag_data, output_dir='analysis_results', storm_periods=None):
    """Plot magnetometer data with highlighted geomagnetic storm periods"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Filter storms to include only those in the relevant time range
    data_start = mag_data.index.min()
    data_end = mag_data.index.max()
    
    # Get timezone information from the data timestamps
    data_tz = data_start.tz
    
    # If no storm periods provided, use historical data
    if storm_periods is None or len(storm_periods) == 0:
        print("No NOAA storm data, using historical storm records")
        storm_periods = []
        for start_date, end_date, description in HISTORICAL_STORMS:
            # Parse dates and add timezone info matching data
            start = pd.to_datetime(start_date).replace(tzinfo=data_tz)
            # Add 1 day to end date to include the full day
            end = (pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)).replace(tzinfo=data_tz)
            
            # Only include storms that overlap with our data timeframe
            if start <= data_end and end >= data_start:
                storm_periods.append((start, end, description))
    else:
        # Ensure all storm periods have the right timezone
        storm_periods = [(start.replace(tzinfo=data_tz) if start.tzinfo is None else start, 
                         end.replace(tzinfo=data_tz) if end.tzinfo is None else end, 
                         desc) for start, end, desc in storm_periods]
    
    if not storm_periods:
        print(f"No geomagnetic storms in the data time period ({data_start} to {data_end})")
    else:
        print(f"Plotting {len(storm_periods)} geomagnetic storms in the data time period:")
        for start, end, desc in storm_periods:
            print(f"  * {desc}: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
    
    devices = mag_data['device_id'].unique()
    
    for device in devices:
        device_data = mag_data[mag_data['device_id'] == device]
        
        # Check if there's data for this device
        if device_data.empty:
            continue
        
        # Determine components to plot based on available data
        components = []
        if 'x' in device_data.columns and not device_data['x'].isna().all() and (device_data['x'] != 0).any():
            components.append(('x', 'red', 'X Component'))
        if 'y' in device_data.columns and not device_data['y'].isna().all() and (device_data['y'] != 0).any():
            components.append(('y', 'green', 'Y Component'))
        if 'z' in device_data.columns and not device_data['z'].isna().all() and (device_data['z'] != 0).any():
            components.append(('z', 'blue', 'Z Component'))
            
        # For the next subplot, choose intensity or field_strength
        if 'intensity' in device_data.columns and not device_data['intensity'].isna().all():
            components.append(('intensity', 'purple', 'Magnetic Intensity'))
        elif 'field_strength' in device_data.columns and not device_data['field_strength'].isna().all():
            components.append(('field_strength', 'purple', 'Field Strength'))
            
        # Add a final subplot for NOAA Kp index if available
        include_kp_plot = False
        kp_data = fetch_noaa_data()
        if kp_data is not None:
            # Ensure consistent timezone for comparison
            utc = pytz.UTC
            
            # Make sure both timestamps have timezone info before comparing
            kp_start = data_start
            kp_end = data_end
            
            # Filter to data range matching our magnetometer data
            try:
                kp_in_range = kp_data[(kp_data.index >= kp_start) & (kp_data.index <= kp_end)]
                if not kp_in_range.empty:
                    include_kp_plot = True
            except TypeError as e:
                print(f"Timezone error in Kp data filtering: {e}")
                # Try again with localized timestamps if needed
                print("Attempting to fix timezone comparison...")
                
                # Convert kp_data index to UTC if it's timezone-naive
                if kp_data.index.tzinfo is None:
                    kp_data.index = kp_data.index.tz_localize(utc)
                
                # Now try the comparison again
                kp_in_range = kp_data[(kp_data.index >= kp_start) & (kp_data.index <= kp_end)]
                if not kp_in_range.empty:
                    include_kp_plot = True
                
        if not components:
            print(f"No plottable data for device {device}")
            continue
            
        # Create figure with subplots
        num_plots = len(components) + (1 if include_kp_plot else 0)
        fig, axes = plt.subplots(num_plots, 1, figsize=(15, 5*num_plots), sharex=True)
        
        # Handle the case where there's only one subplot
        if num_plots == 1:
            axes = [axes]
        
        # Plot components
        for i, (component, color, label) in enumerate(components):
            axes[i].plot(device_data.index, device_data[component], color=color, label=component.upper())
            axes[i].set_ylabel(label)
            
            if i == 0:
                axes[i].set_title(f'Magnetometer Data with Geomagnetic Storm Periods - Device {device}')
            
            # Highlight storm periods
            for start, end, desc in storm_periods:
                # Only highlight if the storm period overlaps with our data
                if not ((end < device_data.index.min()) or (start > device_data.index.max())):
                    axes[i].axvspan(start, end, alpha=0.2, color='yellow')
                    
                    # Add annotation for the first subplot only
                    if i == 0:
                        mid_point = start + (end - start) / 2
                        axes[i].annotate(desc, (mid_point, axes[i].get_ylim()[1] * 0.9),
                                       ha='center', fontsize=9, 
                                       bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
            
            # Add legend
            axes[i].legend(loc='best')
            axes[i].grid(True)
            
        # Add Kp index plot if available
        if include_kp_plot:
            i = len(components)
            kp_ax = axes[i]
            
            # Plot Kp index
            kp_ax.plot(kp_in_range.index, kp_in_range['kp_index'], 'k-', label='Kp Index')
            kp_ax.set_ylabel('Kp Index')
            
            # Add horizontal lines for storm thresholds
            for level, threshold in G_STORM_LEVELS.items():
                kp_ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
                # Only add text if we have data points
                if not kp_in_range.empty:
                    kp_ax.text(kp_in_range.index[0], threshold + 0.1, level, color='r')
                else:
                    # Use axis limits instead if no data
                    kp_ax.text(kp_ax.get_xlim()[0], threshold + 0.1, level, color='r')
            
            # Highlight storm periods
            for start, end, desc in storm_periods:
                if not ((end < kp_in_range.index.min()) or (start > kp_in_range.index.max())):
                    kp_ax.axvspan(start, end, alpha=0.2, color='yellow')
            
            kp_ax.grid(True)
            kp_ax.legend(loc='best')
        
        # Format x-axis
        plt.xlabel('Date')
        fig.autofmt_xdate()
        
        # Use date formatter for x-axis
        date_format = mdates.DateFormatter('%Y-%m-%d')
        plt.gca().xaxis.set_major_formatter(date_format)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/storms_comparison_{device}.png")
        plt.close()
        
        # Create a separate plot just for the Kp index over time if available
        if kp_data is not None:
            try:
                # Try to ensure kp_in_range exists and is not empty
                if not 'kp_in_range' in locals() or kp_in_range.empty:
                    # Ensure consistent timezone for comparison
                    utc = pytz.UTC
                    
                    # Make sure both timestamps have timezone info before comparing
                    kp_start = data_start
                    kp_end = data_end
                    
                    # Convert kp_data index to UTC if it's timezone-naive
                    if kp_data.index.tzinfo is None:
                        kp_data.index = kp_data.index.tz_localize(utc)
                    
                    # Now try the comparison again
                    kp_in_range = kp_data[(kp_data.index >= kp_start) & (kp_data.index <= kp_end)]
                
                if not kp_in_range.empty:
                    plt.figure(figsize=(15, 6))
                    
                    plt.plot(kp_in_range.index, kp_in_range['kp_index'], 'k-', label='Kp Index')
                    plt.ylabel('Kp Index')
                    plt.xlabel('Date')
                else:
                    print("No Kp data found within the data time range. Skipping Kp plot.")
                    continue  # Skip the rest of the plot creation for empty data
            except Exception as e:
                print(f"Error creating Kp index plot: {e}")
                return
            
            # Add horizontal lines for storm thresholds
            for level, threshold in G_STORM_LEVELS.items():
                plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
                # Only add text if we have data points
                if not kp_in_range.empty:
                    plt.text(kp_in_range.index[0], threshold + 0.1, level, color='r')
                else:
                    # Use a reasonable fallback for text position
                    plt.text(plt.xlim()[0], threshold + 0.1, level, color='r')
            
            # Highlight storm periods
            for start, end, desc in storm_periods:
                if not ((end < kp_in_range.index.min()) or (start > kp_in_range.index.max())):
                    plt.axvspan(start, end, alpha=0.2, color='yellow')
                    # Add annotation
                    mid_point = start + (end - start) / 2
                    plt.annotate(desc, (mid_point, plt.ylim()[1] * 0.9),
                               ha='center', fontsize=9, 
                               bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
            
            plt.title('NOAA Planetary Kp Index')
            plt.grid(True)
            plt.legend(loc='best')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/kp_index_timeseries.png")
            plt.close()

def analyze_during_storms(mag_data, output_dir='analysis_results', storm_periods=None):
    """Analyze magnetometer readings specifically during known storm periods"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Filter storms to include only those in the relevant time range
    data_start = mag_data.index.min()
    data_end = mag_data.index.max()
    
    # Get timezone information from the data timestamps
    data_tz = data_start.tz
    
    # If no storm periods provided, use historical data
    if storm_periods is None or len(storm_periods) == 0:
        print("No NOAA storm data, using historical storm records for analysis")
        storm_periods = []
        for start_date, end_date, description in HISTORICAL_STORMS:
            # Parse dates and add timezone info matching data
            start = pd.to_datetime(start_date).replace(tzinfo=data_tz)
            # Add 1 day to end date to include the full day
            end = (pd.to_datetime(end_date) + timedelta(days=1) - timedelta(seconds=1)).replace(tzinfo=data_tz)
            
            # Only include storms that overlap with our data timeframe
            if start <= data_end and end >= data_start:
                storm_periods.append((start, end, description))
    else:
        # Ensure all storm periods have the right timezone
        storm_periods = [(start.replace(tzinfo=data_tz) if start.tzinfo is None else start, 
                         end.replace(tzinfo=data_tz) if end.tzinfo is None else end, 
                         desc) for start, end, desc in storm_periods]
    
    if not storm_periods:
        print("No storms to analyze within the data time period")
        return None
    
    devices = mag_data['device_id'].unique()
    
    # Create a DataFrame to store statistics
    storm_stats = []
    
    for device in devices:
        device_data = mag_data[mag_data['device_id'] == device]
        
        # Check if there's data for this device
        if device_data.empty:
            continue
            
        # Determine which field to analyze: intensity or field_strength
        analysis_field = None
        if 'intensity' in device_data.columns and not device_data['intensity'].isna().all():
            analysis_field = 'intensity'
        elif 'field_strength' in device_data.columns and not device_data['field_strength'].isna().all():
            analysis_field = 'field_strength'
        else:
            print(f"No intensity or field_strength data for device {device}, skipping storm analysis")
            continue
            
        # Get baseline stats for non-storm periods
        storm_mask = pd.Series(False, index=device_data.index)
        for start, end, _ in storm_periods:
            period_mask = (device_data.index >= start) & (device_data.index <= end)
            storm_mask = storm_mask | period_mask
            
        baseline_data = device_data[~storm_mask]
        
        if not baseline_data.empty:
            baseline_stats = {
                f'{analysis_field}_mean': baseline_data[analysis_field].mean(),
                f'{analysis_field}_std': baseline_data[analysis_field].std(),
            }
            
            # Add component stats if available
            for component in ['x', 'y', 'z']:
                if component in baseline_data.columns and not baseline_data[component].isna().all() and (baseline_data[component] != 0).any():
                    baseline_stats[f'{component}_mean'] = baseline_data[component].mean()
                    baseline_stats[f'{component}_std'] = baseline_data[component].std()
        else:
            # If no baseline data available, skip this device
            continue
            
        # Analyze each storm period
        for start, end, description in storm_periods:
            # Get data for this storm period
            period_data = device_data[(device_data.index >= start) & (device_data.index <= end)]
            
            if len(period_data) > 0:
                # Calculate statistics
                field_mean = period_data[analysis_field].mean()
                field_std = period_data[analysis_field].std()
                
                # Calculate Z-scores (how many standard deviations from baseline)
                field_zscore = (field_mean - baseline_stats[f'{analysis_field}_mean']) / baseline_stats[f'{analysis_field}_std']
                
                component_zscores = {}
                for component in ['x', 'y', 'z']:
                    component_key = f'{component}_mean'
                    std_key = f'{component}_std'
                    if component_key in baseline_stats and std_key in baseline_stats and component in period_data.columns:
                        # Only calculate if the component has actual values
                        if not period_data[component].isna().all() and (period_data[component] != 0).any():
                            component_zscores[f'{component}_zscore'] = (
                                (period_data[component].mean() - baseline_stats[component_key]) / 
                                baseline_stats[std_key]
                            )
                
                # Store statistics
                stats_entry = {
                    'device_id': device,
                    'storm_description': description,
                    'start_date': start.date(),
                    'end_date': end.date(),
                    'data_points': len(period_data),
                    f'{analysis_field}_mean': field_mean,
                    f'{analysis_field}_std': field_std,
                    f'{analysis_field}_zscore': field_zscore
                }
                
                # Add component z-scores
                stats_entry.update(component_zscores)
                
                # Calculate maximum deviation across components
                deviations = [abs(z) for k, z in component_zscores.items()]
                if deviations:
                    stats_entry['max_deviation'] = max([abs(field_zscore)] + deviations)
                else:
                    stats_entry['max_deviation'] = abs(field_zscore)
                    
                storm_stats.append(stats_entry)
    
    # Create DataFrame from collected stats
    if storm_stats:
        stats_df = pd.DataFrame(storm_stats)
        
        # Save statistics to CSV
        stats_df.to_csv(f"{output_dir}/storm_analysis_stats.csv", index=False)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        sns.barplot(x='storm_description', y='max_deviation', hue='device_id', data=stats_df)
        plt.title('Maximum Deviation During Geomagnetic Storms')
        plt.xlabel('Storm Event')
        plt.ylabel('Maximum Deviation (Z-score)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/storm_deviations.png")
        plt.close()
        
        return stats_df
    else:
        print("No storm data analysis available")
        return None

def main():
    # Load processed data
    mag_data = load_processed_data()
    
    if mag_data is None:
        print("Failed to load processed data. Run the analyze_magnetometer_data.py script first.")
        return
    
    output_dir = 'analysis_results'
    
    # Fetch NOAA data and identify storm periods
    kp_data = fetch_noaa_data()
    storm_periods = []
    
    if kp_data is not None:
        storm_periods = identify_storm_periods(kp_data)
    
    # Plot data with storm periods highlighted
    plot_data_with_storm_periods(mag_data, output_dir, storm_periods)
    
    # Analyze data during storm periods
    storm_stats = analyze_during_storms(mag_data, output_dir, storm_periods)
    
    if storm_stats is not None:
        # Print summary of storm analysis
        print("\nStorm Analysis Summary:")
        print("=======================")
        print(f"Analyzed {len(storm_stats['storm_description'].unique())} geomagnetic storm periods")
        
        # Find storm periods with significant deviations (Z-score > 2)
        significant = storm_stats[storm_stats['max_deviation'] > 2]
        if not significant.empty:
            print(f"\nSignificant deviations found in {len(significant)} storm periods:")
            for _, row in significant.iterrows():
                print(f"  * {row['storm_description']} ({row['start_date']} to {row['end_date']}): "
                      f"Device {row['device_id']} - Max deviation: {row['max_deviation']:.2f} σ")
        else:
            print("\nNo significant deviations found during known storm periods.")
            
        print(f"\nDetailed results saved to {output_dir}/storm_analysis_stats.csv")

if __name__ == "__main__":
    main() 
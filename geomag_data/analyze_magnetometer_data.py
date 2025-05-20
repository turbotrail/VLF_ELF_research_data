import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
from scipy import signal
from scipy.stats import zscore
import os

# Configure better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Function to load the data
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    # For extremely large files, we need to read in chunks
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Print the structure of the first few records for debugging
    print("Sample data structure:")
    for i, item in enumerate(data[:3]):
        print(f"Record {i+1}:")
        for key, value in item.items():
            print(f"  {key}: {type(value)} - {str(value)[:100]}")
        print()
    
    print(f"Data loaded successfully. Processing...")
    return data

# Function to convert MongoDB data to pandas DataFrame
def process_data(data):
    # The structure depends on the actual format of your JSON
    # This is a placeholder - adjust according to actual structure
    records = []
    
    for item in data:
        # Extract relevant fields
        # Adjust these field names according to your data structure
        try:
            timestamp = item.get('timestamp')
            device_id = item.get('device_id')
            
            # Extract magnetometer values safely with conversion to float
            # Some records have direct x, y, z fields
            x = float(item.get('x', 0)) if item.get('x') is not None else 0.0
            y = float(item.get('y', 0)) if item.get('y') is not None else 0.0
            z = float(item.get('z', 0)) if item.get('z') is not None else 0.0
            
            # Some records have nested magnetometer object
            magnetometer = item.get('magnetometer', {})
            if isinstance(magnetometer, dict) and any(magnetometer.values()):
                if magnetometer.get('x') is not None:
                    x = float(magnetometer.get('x', 0))
                if magnetometer.get('y') is not None:
                    y = float(magnetometer.get('y', 0))
                if magnetometer.get('z') is not None:
                    z = float(magnetometer.get('z', 0))
            
            # Some records have field_strength directly
            field_strength = None
            if item.get('field_strength') is not None:
                field_strength = float(item.get('field_strength'))
            
            # Temperature data if available
            temperature = float(item.get('temperature', 0)) if item.get('temperature') is not None else None
            
            # Convert timestamp to datetime based on its type
            if isinstance(timestamp, dict):
                # Handle MongoDB Date format which might be stored as a dictionary
                # with keys like $date or $numberLong
                if '$date' in timestamp:
                    # Could be ISO string or milliseconds
                    date_value = timestamp['$date']
                    if isinstance(date_value, str):
                        timestamp = pd.to_datetime(date_value)
                    elif isinstance(date_value, (int, float)):
                        # Milliseconds since epoch
                        timestamp = pd.to_datetime(date_value, unit='ms')
                elif '$numberLong' in timestamp:
                    # MongoDB NumberLong type
                    timestamp = pd.to_datetime(int(timestamp['$numberLong']), unit='ms')
            elif isinstance(timestamp, str):
                # Try to parse ISO format
                timestamp = pd.to_datetime(timestamp)
            elif isinstance(timestamp, (int, float)):
                # Assume milliseconds since epoch
                timestamp = pd.to_datetime(timestamp, unit='ms')
            
            # Skip record if we can't parse the timestamp
            if timestamp is None:
                print(f"Skipping record with unparseable timestamp: {item.get('timestamp')}")
                continue
            
            record = {
                'timestamp': timestamp,
                'device_id': device_id,
                'x': x,
                'y': y,
                'z': z,
                'temperature': temperature
            }
            
            # Add field_strength if it exists
            if field_strength is not None:
                record['field_strength'] = field_strength
                
                # If we have field_strength but no components, try to derive them
                if x == 0 and y == 0 and z == 0:
                    # We can't know the exact components, but we can make a reasonable guess
                    # by assigning the field strength to one component (for visualization purposes)
                    record['x'] = field_strength
            
            records.append(record)
        except Exception as e:
            print(f"Error processing record: {e}")
            # For debugging, print a sample of the problematic record
            print(f"Problematic record sample: {str(item)[:200]}...")
            continue
    
    df = pd.DataFrame(records)
    
    # Skip empty dataframe
    if df.empty:
        print("No valid records found. Please check your data format.")
        return df
    
    # Set timestamp as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Print DataFrame info for debugging
    print("\nDataFrame summary:")
    print(f"Records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:")
    print(df.dtypes)
    print("\nFirst few records:")
    print(df.head())
    
    return df

# Function to calculate magnetic intensity
def calculate_magnetic_intensity(df):
    if df.empty:
        return df
    
    try:
        # Ensure x, y, z are numeric
        for col in ['x', 'y', 'z']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate intensity using element-wise operations
        df['intensity'] = np.sqrt(df['x'].pow(2) + df['y'].pow(2) + df['z'].pow(2))
        
        # If field_strength exists but intensity doesn't make sense, use field_strength
        if 'field_strength' in df.columns:
            # Where intensity is NaN or 0, use field_strength
            mask = (df['intensity'].isna()) | (df['intensity'] == 0)
            df.loc[mask, 'intensity'] = df.loc[mask, 'field_strength']
        
        return df
    except Exception as e:
        print(f"Error calculating intensity: {e}")
        # Return original dataframe without intensity
        return df

# Function to detect anomalies using z-score
def detect_anomalies(df, column, threshold=3):
    if df.empty or column not in df.columns:
        return df
    
    try:
        # Only calculate z-score on numeric values
        numeric_data = pd.to_numeric(df[column], errors='coerce')
        df[f'{column}_zscore'] = zscore(numeric_data, nan_policy='omit')
        df[f'{column}_anomaly'] = df[f'{column}_zscore'].abs() > threshold
    except Exception as e:
        print(f"Error detecting anomalies: {e}")
    
    return df

# Function to plot time series data
def plot_time_series(df, title, output_dir='plots'):
    if df.empty:
        print("No data to plot.")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot by device
    devices = df['device_id'].unique()
    
    for device in devices:
        device_data = df[df['device_id'] == device]
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
        
        # Plot X, Y, Z components
        axes[0].plot(device_data.index, device_data['x'], color='red', label='X')
        axes[0].set_ylabel('X Component')
        axes[0].set_title(f'Magnetometer Data - Device {device}')
        
        axes[1].plot(device_data.index, device_data['y'], color='green', label='Y')
        axes[1].set_ylabel('Y Component')
        
        axes[2].plot(device_data.index, device_data['z'], color='blue', label='Z')
        axes[2].set_ylabel('Z Component')
        
        # Plot magnetic intensity
        if 'intensity' in device_data.columns:
            axes[3].plot(device_data.index, device_data['intensity'], color='purple', label='Intensity')
            axes[3].set_ylabel('Magnetic Intensity')
            
            # Highlight anomalies if they exist
            if 'intensity_anomaly' in device_data.columns:
                anomalies = device_data[device_data['intensity_anomaly']]
                if not anomalies.empty:
                    axes[3].scatter(anomalies.index, anomalies['intensity'], 
                                  color='red', marker='o', s=50, label='Anomalies')
        elif 'field_strength' in device_data.columns:
            axes[3].plot(device_data.index, device_data['field_strength'], color='purple', label='Field Strength')
            axes[3].set_ylabel('Field Strength')
        
        # Format x-axis
        plt.xlabel('Date')
        fig.autofmt_xdate()
        
        # Add legend to each subplot
        for ax in axes:
            ax.legend(loc='best')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{title}_{device}.png")
        plt.close()

# Function to perform spectral analysis
def spectral_analysis(df, output_dir='plots'):
    if df.empty:
        print("No data for spectral analysis.")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    devices = df['device_id'].unique()
    
    for device in devices:
        try:
            device_data = df[df['device_id'] == device].copy()
            
            # Keep only numeric columns for resampling
            numeric_cols = device_data.select_dtypes(include=np.number).columns.tolist()
            
            # Ensure data is regularly sampled - only resample numeric columns
            if numeric_cols:
                # First, create a copy with just the device_id and numeric columns
                device_numeric = device_data[numeric_cols].copy()
                
                # Resample the numeric data
                device_numeric = device_numeric.resample('1H').mean()
                
                # Fill NaN values with forward fill, then backward fill
                device_numeric = device_numeric.fillna(method='ffill').fillna(method='bfill')
                
                if device_numeric.empty:
                    print(f"No data for device {device} after resampling")
                    continue
                
                # Create plot
                fig, axes = plt.subplots(min(4, len(numeric_cols)), 1, figsize=(15, 20))
                if len(numeric_cols) == 1:
                    axes = [axes]  # Make it a list for consistent indexing
                
                # Plot each numeric column
                for i, column in enumerate(['x', 'y', 'z', 'intensity']):
                    if i < len(axes) and column in device_numeric.columns:
                        # Apply FFT
                        data = device_numeric[column].values
                        data = data - np.mean(data)  # Remove DC component
                        
                        try:
                            # Calculate sampling frequency (samples per day)
                            if len(device_numeric) > 1:
                                time_diff = device_numeric.index[1] - device_numeric.index[0]
                                fs = pd.Timedelta('1D') / time_diff
                                
                                # Compute periodogram
                                f, Pxx = signal.periodogram(data, fs=fs)
                                
                                # Plot in period domain (days) rather than frequency
                                period = 1/f[1:]  # Skip first point (DC component)
                                power = Pxx[1:]
                                
                                axes[i].loglog(period, power)
                                axes[i].set_xlabel('Period (days)')
                                axes[i].set_ylabel('Power')
                                axes[i].set_title(f'{column.upper()} Component - Spectral Analysis')
                                axes[i].grid(True)
                                
                                # Mark significant periods
                                peak_indices = signal.find_peaks(power)[0]
                                for peak_idx in peak_indices:
                                    if power[peak_idx] > 0.8 * np.max(power):  # Only label significant peaks
                                        axes[i].axvline(x=period[peak_idx], color='r', linestyle='--', alpha=0.5)
                                        axes[i].text(period[peak_idx], power[peak_idx], 
                                                  f"{period[peak_idx]:.1f} days", 
                                                  rotation=90, verticalalignment='bottom')
                            else:
                                axes[i].text(0.5, 0.5, 'Not enough data points for spectral analysis',
                                           ha='center', va='center', transform=axes[i].transAxes)
                        except Exception as e:
                            print(f"Error in spectral analysis for {column}: {e}")
                            axes[i].text(0.5, 0.5, f'Error: {str(e)}',
                                       ha='center', va='center', transform=axes[i].transAxes)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/spectral_analysis_{device}.png")
                plt.close()
            else:
                print(f"No numeric data for device {device}")
        except Exception as e:
            print(f"Error in spectral analysis for device {device}: {e}")

# Function to correlate with temperature
def correlation_analysis(df, output_dir='plots'):
    if df.empty or 'temperature' not in df.columns:
        print("No temperature data available for correlation analysis.")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    devices = df['device_id'].unique()
    
    for device in devices:
        device_data = df[df['device_id'] == device]
        
        # Only proceed if temperature data exists and is not all NaN
        if 'temperature' in device_data.columns and not device_data['temperature'].isnull().all():
            try:
                # Keep only numeric columns
                numeric_cols = device_data.select_dtypes(include=np.number).columns.tolist()
                
                # Make sure temperature is in the numeric columns
                if 'temperature' not in numeric_cols:
                    print(f"Temperature is not numeric for device {device}")
                    continue
                
                # Calculate correlation matrix
                corr_matrix = device_data[numeric_cols].corr()
                
                # Plot correlation heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Correlation Matrix - Device {device}')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/correlation_matrix_{device}.png")
                plt.close()
                
                # Scatter plots for temperature vs magnetic components
                components = [c for c in ['x', 'y', 'z', 'intensity', 'field_strength'] if c in numeric_cols]
                if components:
                    fig, axes = plt.subplots(min(2, len(components)), 
                                          min(2, (len(components) + 1) // 2), 
                                          figsize=(15, 12))
                    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]
                    
                    for i, component in enumerate(components):
                        if i < len(axes):
                            sns.scatterplot(x='temperature', y=component, data=device_data, ax=axes[i], alpha=0.5)
                            
                            # Add regression line
                            sns.regplot(x='temperature', y=component, data=device_data, 
                                       ax=axes[i], scatter=False, color='red')
                            
                            axes[i].set_title(f'Temperature vs {component.upper()} - Device {device}')
                    
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/temp_vs_magnetic_{device}.png")
                    plt.close()
            except Exception as e:
                print(f"Error in correlation analysis for device {device}: {e}")

def main():
    # Define file paths
    collection_file = 'collection_data.json'
    
    try:
        # Load data
        raw_data = load_data(collection_file)
        
        # Process data
        df = process_data(raw_data)
        
        if df.empty:
            print("No valid data to analyze. Please check your data format.")
            return
            
        # Calculate magnetic intensity
        df = calculate_magnetic_intensity(df)
        
        # Detect anomalies
        if 'intensity' in df.columns:
            df = detect_anomalies(df, 'intensity')
        elif 'field_strength' in df.columns:
            df = detect_anomalies(df, 'field_strength')
        
        # Create plots directory
        output_dir = 'analysis_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save processed data - with consistent timestamp format
        # Convert timestamp to string in a standard format that can be easily parsed
        df_to_save = df.reset_index()
        df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f%z')
        df_to_save.to_csv(f"{output_dir}/processed_data.csv", index=False)
        print(f"Processed data saved to {output_dir}/processed_data.csv")
        
        # Plot time series
        plot_time_series(df, 'magnetometer_data', output_dir)
        
        # Perform spectral analysis
        spectral_analysis(df, output_dir)
        
        # Correlation analysis with temperature
        correlation_analysis(df, output_dir)
        
        print(f"Analysis complete. Results saved in {output_dir} directory.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
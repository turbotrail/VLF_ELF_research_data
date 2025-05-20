import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
import math
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_processed_data(file_path='analysis_results/processed_data.csv'):
    """Load the previously processed magnetometer data"""
    print(f"Loading processed data from {file_path}...")
    try:
        # Use flexible datetime parser to handle various formats
        df = pd.read_csv(file_path, parse_dates=['timestamp'], infer_datetime_format=True)
        df = df.set_index('timestamp')
        
        # Print information about the loaded data
        print(f"Successfully loaded {len(df)} records spanning from {df.index.min()} to {df.index.max()}")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Check which devices have three-axis data
        devices = df['device_id'].unique()
        for device in devices:
            device_data = df[df['device_id'] == device]
            has_xyz = all(col in device_data.columns for col in ['x', 'y', 'z'])
            has_values = False
            if has_xyz:
                has_values = not (device_data['x'].isna().all() or 
                                 device_data['y'].isna().all() or 
                                 device_data['z'].isna().all())
            
            if has_xyz and has_values:
                print(f"Device {device}: Has valid three-axis (x,y,z) data")
            else:
                if 'field_strength' in device_data.columns:
                    print(f"Device {device}: Has field_strength data only")
                else:
                    print(f"Device {device}: Missing critical magnetic data")
        
        return df
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

def check_device_has_xyz(device_data):
    """Check if a device has valid xyz data"""
    has_xyz = all(col in device_data.columns for col in ['x', 'y', 'z'])
    if not has_xyz:
        return False
    
    # Check if the data is actually valid (not all zeros or NaN)
    has_values = not (device_data['x'].isna().all() or 
                     device_data['y'].isna().all() or 
                     device_data['z'].isna().all())
    
    # Also check if we don't have all zeros
    if has_values:
        all_zeros = ((device_data['x'] == 0) & 
                     (device_data['y'] == 0) & 
                     (device_data['z'] == 0)).all()
        return not all_zeros
    
    return has_values

def calculate_orientation_angles(df):
    """Calculate magnetic field orientation angles from XYZ components"""
    print("Calculating magnetic orientation angles...")
    
    # Process each device separately
    devices = df['device_id'].unique()
    for device in devices:
        device_data = df[df['device_id'] == device]
        
        # Check if this device has xyz data
        if check_device_has_xyz(device_data):
            print(f"Calculating orientation angles for device {device} (has xyz data)")
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            
            # Declination angle (angle between magnetic north and true north in xy-plane)
            df.loc[df['device_id'] == device, 'declination'] = np.degrees(
                np.arctan2(device_data['y'], device_data['x'])
            )
            
            # Inclination angle (angle between magnetic field vector and horizontal plane)
            horizontal_intensity = np.sqrt(device_data['x']**2 + device_data['y']**2)
            df.loc[df['device_id'] == device, 'inclination'] = np.degrees(
                np.arctan2(device_data['z'], horizontal_intensity + epsilon)
            )
        else:
            print(f"Skipping orientation angle calculation for device {device} (no xyz data)")
    
    return df

def smooth_time_series(df, column, window_length=31, polyorder=3):
    """Apply Savitzky-Golay filter to smooth time series data"""
    # We need to handle each device separately
    devices = df['device_id'].unique()
    df_smoothed = df.copy()
    
    for device in devices:
        device_mask = df['device_id'] == device
        
        # Skip if column doesn't exist for this device
        if column not in df.loc[device_mask].columns:
            continue
            
        # Only apply smoothing if we have enough data points
        if sum(device_mask) > window_length:
            try:
                df_smoothed.loc[device_mask, f'{column}_smooth'] = savgol_filter(
                    df.loc[device_mask, column].values, 
                    window_length, 
                    polyorder
                )
            except Exception as e:
                print(f"Error smoothing {column} for device {device}: {e}")
                # Just copy the original data if smoothing fails
                df_smoothed.loc[device_mask, f'{column}_smooth'] = df.loc[device_mask, column]
        else:
            # Just copy the original data if not enough points
            df_smoothed.loc[device_mask, f'{column}_smooth'] = df.loc[device_mask, column]
    
    return df_smoothed

def plot_magnetic_orientation(df, output_dir='analysis_results'):
    """Plot magnetic field orientation trends over time"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    devices = df['device_id'].unique()
    
    for device in devices:
        device_data = df[df['device_id'] == device]
        
        # Check if this device has orientation angles
        if 'declination' not in device_data.columns or 'inclination' not in device_data.columns:
            print(f"Device {device} doesn't have orientation angles, skipping orientation plots")
            continue
            
        # Check if there's data for this device
        if len(device_data) == 0:
            continue
            
        # Resample to daily averages to reduce noise
        try:
            # Only use numeric columns for resampling
            numeric_cols = device_data.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                print(f"No numeric data for device {device}")
                continue
                
            # Make a copy with just timestamp and numeric data
            daily_data = device_data[numeric_cols].resample('D').mean()
                
            # Add device_id back
            daily_data['device_id'] = device
            
            # Apply smoothing to declination and inclination
            if 'declination' in daily_data.columns and 'inclination' in daily_data.columns:
                daily_data = smooth_time_series(daily_data.reset_index(), 'declination')
                daily_data = smooth_time_series(daily_data, 'inclination')
                daily_data = daily_data.set_index('timestamp')
                
                # Create a figure with two subplots
                fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Plot declination angle
                axes[0].plot(daily_data.index, daily_data['declination'], 'b-', alpha=0.5, label='Raw')
                axes[0].plot(daily_data.index, daily_data['declination_smooth'], 'b-', label='Smoothed')
                axes[0].set_ylabel('Declination (degrees)')
                axes[0].set_title(f'Magnetic Declination - Device {device}')
                axes[0].grid(True)
                axes[0].legend()
                
                # Plot inclination angle
                axes[1].plot(daily_data.index, daily_data['inclination'], 'r-', alpha=0.5, label='Raw')
                axes[1].plot(daily_data.index, daily_data['inclination_smooth'], 'r-', label='Smoothed')
                axes[1].set_ylabel('Inclination (degrees)')
                axes[1].set_title(f'Magnetic Inclination - Device {device}')
                axes[1].grid(True)
                axes[1].legend()
                
                # Format x-axis
                plt.xlabel('Date')
                fig.autofmt_xdate()
                
                # Use date formatter for x-axis
                date_format = mdates.DateFormatter('%Y-%m-%d')
                plt.gca().xaxis.set_major_formatter(date_format)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/magnetic_orientation_{device}.png")
                plt.close()
                
                # Also create a combined plot showing movement over time
                plt.figure(figsize=(10, 8))
                
                # Color points by time for visualization
                time_normalized = (daily_data.index - daily_data.index.min()) / (daily_data.index.max() - daily_data.index.min())
                time_colors = plt.cm.viridis(time_normalized)
                
                # Scatter plot of declination vs. inclination
                sc = plt.scatter(daily_data['declination_smooth'], daily_data['inclination_smooth'], 
                                c=time_normalized, cmap='viridis', s=20, alpha=0.7)
                
                # Connect points in time sequence
                plt.plot(daily_data['declination_smooth'], daily_data['inclination_smooth'], 'k-', alpha=0.3)
                
                # Mark start and end points
                plt.scatter(daily_data['declination_smooth'].iloc[0], daily_data['inclination_smooth'].iloc[0], 
                           color='green', s=100, marker='o', label='Start')
                plt.scatter(daily_data['declination_smooth'].iloc[-1], daily_data['inclination_smooth'].iloc[-1], 
                           color='red', s=100, marker='o', label='End')
                
                plt.xlabel('Declination (degrees)')
                plt.ylabel('Inclination (degrees)')
                plt.title(f'Magnetic Field Orientation Movement - Device {device}')
                plt.grid(True)
                plt.colorbar(sc, label='Time Progression')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/magnetic_movement_{device}.png")
                plt.close()
            else:
                print(f"Declination or inclination data missing for device {device}")
        except Exception as e:
            print(f"Error processing orientation data for device {device}: {e}")

def plot_field_strength_trends(df, output_dir='analysis_results'):
    """Plot field strength trends for devices without xyz data"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    devices = df['device_id'].unique()
    
    for device in devices:
        device_data = df[df['device_id'] == device]
        
        # Skip devices with xyz data (they're handled separately)
        if check_device_has_xyz(device_data):
            continue
            
        # Check if we have field_strength
        if 'field_strength' not in device_data.columns:
            print(f"Device {device} doesn't have field_strength data, skipping trend analysis")
            continue
            
        try:
            # Resample to daily averages
            daily_data = device_data[['field_strength']].resample('D').mean()
            
            # Skip if no data
            if daily_data.empty:
                continue
                
            # Apply smoothing
            daily_df = daily_data.reset_index()
            daily_df['device_id'] = device
            daily_df = smooth_time_series(daily_df, 'field_strength', window_length=7, polyorder=2)
            daily_df = daily_df.set_index('timestamp')
            
            # Plot the field strength over time
            plt.figure(figsize=(12, 6))
            
            # Plot raw and smoothed data
            plt.plot(daily_df.index, daily_df['field_strength'], 'b-', alpha=0.5, label='Raw')
            plt.plot(daily_df.index, daily_df['field_strength_smooth'], 'r-', label='Smoothed (7-day window)')
            
            # Calculate and plot trend line
            x = np.arange(len(daily_df))
            y = daily_df['field_strength'].values
            
            # Handle NaN values
            valid_indices = ~np.isnan(y)
            if sum(valid_indices) >= 2:  # Need at least 2 points for a line
                x_valid = x[valid_indices]
                y_valid = y[valid_indices]
                
                z = np.polyfit(x_valid, y_valid, 1)
                p = np.poly1d(z)
                
                # Calculate slope in units per year
                days_span = (daily_df.index.max() - daily_df.index.min()).days
                if days_span > 0:
                    units_per_day = z[0]
                    units_per_year = units_per_day * 365.25
                    
                    # Calculate R-squared
                    y_pred = p(x_valid)
                    ss_total = np.sum((y_valid - np.mean(y_valid))**2)
                    ss_residual = np.sum((y_valid - y_pred)**2)
                    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                    
                    # Plot trendline
                    plt.plot(daily_df.index, p(x), 'g--', 
                           label=f'Trend: {units_per_year:.2f} units/year (R²={r_squared:.3f})')
            
            plt.title(f'Magnetic Field Strength Trend - Device {device}')
            plt.ylabel('Field Strength')
            plt.xlabel('Date')
            plt.grid(True)
            plt.legend()
            fig = plt.gcf()
            fig.autofmt_xdate()
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/field_strength_trend_{device}.png")
            plt.close()
            
        except Exception as e:
            print(f"Error analyzing field strength for device {device}: {e}")

def detect_trends_and_shifts(df, output_dir='analysis_results'):
    """
    Detect significant trends or shifts in the magnetic orientation data
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    devices = df['device_id'].unique()
    
    trends = []
    
    for device in devices:
        try:
            device_data = df[df['device_id'] == device].copy()
            
            # Skip if not enough data
            if len(device_data) < 30:
                continue
                
            # Check if this device has orientation data
            if not check_device_has_xyz(device_data):
                # Skip orientation analysis for devices without xyz data
                continue
                
            # Resample to daily data
            numeric_cols = device_data.select_dtypes(include=np.number).columns.tolist()
            
            if 'declination' in numeric_cols and 'inclination' in numeric_cols:
                daily_data = device_data[numeric_cols].resample('D').mean()
                
                # For each component (declination and inclination)
                for component in ['declination', 'inclination']:
                    if component not in daily_data.columns:
                        continue
                        
                    # Calculate linear trend
                    x = np.arange(len(daily_data))
                    y = daily_data[component].values
                    
                    # Handle NaN values
                    valid_indices = ~np.isnan(y)
                    if sum(valid_indices) < 10:
                        continue
                        
                    x_valid = x[valid_indices]
                    y_valid = y[valid_indices]
                    
                    # Calculate trend (degrees per day)
                    if len(x_valid) > 1:
                        z = np.polyfit(x_valid, y_valid, 1)
                        slope = z[0]
                        
                        # Annualize the trend (degrees per year)
                        annualized_trend = slope * 365.25
                        
                        # Calculate R-squared to determine trend significance
                        p = np.poly1d(z)
                        y_pred = p(x_valid)
                        ss_total = np.sum((y_valid - np.mean(y_valid))**2)
                        ss_residual = np.sum((y_valid - y_pred)**2)
                        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                        
                        # Store the trend information
                        trends.append({
                            'device_id': device,
                            'component': component,
                            'start_date': daily_data.index.min(),
                            'end_date': daily_data.index.max(),
                            'duration_days': (daily_data.index.max() - daily_data.index.min()).days,
                            'trend_per_day': slope,
                            'trend_per_year': annualized_trend,
                            'r_squared': r_squared
                        })
                        
                        # Plot the trend
                        plt.figure(figsize=(12, 6))
                        plt.plot(daily_data.index, y, 'b.', alpha=0.5)
                        
                        # Plot trendline
                        xd = np.array([daily_data.index.min(), daily_data.index.max()])
                        yd = p(np.array([0, len(daily_data)-1]))
                        plt.plot(xd, yd, 'r-')
                        
                        plt.xlabel('Date')
                        plt.ylabel(f'{component.capitalize()} (degrees)')
                        plt.title(f'{component.capitalize()} Trend - Device {device}\n'
                                 f'Slope: {annualized_trend:.2f}° per year, R²: {r_squared:.3f}')
                        plt.grid(True)
                        fig = plt.gcf()
                        fig.autofmt_xdate()
                        plt.tight_layout()
                        plt.savefig(f"{output_dir}/{component}_trend_{device}.png")
                        plt.close()
        except Exception as e:
            print(f"Error detecting trends for device {device}: {e}")
    
    if trends:
        # Create DataFrame from trends
        trends_df = pd.DataFrame(trends)
        
        # Save trends to CSV
        trends_df.to_csv(f"{output_dir}/magnetic_trends.csv", index=False)
        
        # Create summary plot for all devices
        plt.figure(figsize=(12, 8))
        
        # Plot declination trends
        declination_trends = trends_df[trends_df['component'] == 'declination']
        if not declination_trends.empty:
            plt.subplot(2, 1, 1)
            for device in declination_trends['device_id'].unique():
                device_trend = declination_trends[declination_trends['device_id'] == device]
                plt.bar(device, device_trend['trend_per_year'].values[0], 
                       alpha=0.7, yerr=abs(device_trend['trend_per_year'].values[0]) * (1-device_trend['r_squared'].values[0]))
            
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.ylabel('Degrees per Year')
            plt.title('Declination Trend by Device')
            plt.grid(True, axis='y')
        
        # Plot inclination trends
        inclination_trends = trends_df[trends_df['component'] == 'inclination']
        if not inclination_trends.empty:
            plt.subplot(2, 1, 2)
            for device in inclination_trends['device_id'].unique():
                device_trend = inclination_trends[inclination_trends['device_id'] == device]
                plt.bar(device, device_trend['trend_per_year'].values[0], 
                       alpha=0.7, yerr=abs(device_trend['trend_per_year'].values[0]) * (1-device_trend['r_squared'].values[0]))
            
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.ylabel('Degrees per Year')
            plt.title('Inclination Trend by Device')
            plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/magnetic_trends_summary.png")
        plt.close()
        
        return trends_df
    else:
        print("No significant trends detected")
        return None

def main():
    # Load processed data
    df = load_processed_data()
    
    if df is None:
        print("Failed to load processed data. Run the analyze_magnetometer_data.py script first.")
        return
    
    output_dir = 'analysis_results'
    
    # Calculate orientation angles (only for devices with xyz data)
    df = calculate_orientation_angles(df)
    
    # Plot magnetic orientation over time (for devices with xyz data)
    plot_magnetic_orientation(df, output_dir)
    
    # Plot field strength trends (for devices without xyz data)
    plot_field_strength_trends(df, output_dir)
    
    # Detect trends and shifts in the orientation data (for devices with xyz data)
    trends_df = detect_trends_and_shifts(df, output_dir)
    
    if trends_df is not None and not trends_df.empty:
        # Print summary of significant trends
        significant_trends = trends_df[trends_df['r_squared'] > 0.5]
        if not significant_trends.empty:
            print("\nSignificant Magnetic Field Orientation Trends:")
            print("===========================================")
            
            for _, row in significant_trends.iterrows():
                print(f"Device {row['device_id']} - {row['component'].capitalize()}:")
                print(f"  Trend: {row['trend_per_year']:.4f}° per year (R² = {row['r_squared']:.3f})")
                print(f"  Period: {row['start_date'].date()} to {row['end_date'].date()} ({row['duration_days']} days)")
                print()
            
            avg_dec_trend = significant_trends[significant_trends['component'] == 'declination']['trend_per_year'].mean()
            avg_inc_trend = significant_trends[significant_trends['component'] == 'inclination']['trend_per_year'].mean()
            
            print("Average Trends:")
            if not np.isnan(avg_dec_trend):
                print(f"  Declination: {avg_dec_trend:.4f}° per year")
            if not np.isnan(avg_inc_trend):
                print(f"  Inclination: {avg_inc_trend:.4f}° per year")
                
            # Compare with published rates of magnetic pole movement
            print("\nContext: The Earth's magnetic north pole has been moving at about 55-60 km per year")
            print("        or approximately 0.3-0.5 degrees per year in recent decades.")
        else:
            print("\nNo statistically significant trends detected in the magnetic orientation data.")
    else:
        print("\nNo orientation trends were detected or no devices had valid xyz data for orientation analysis.")
    
    print(f"\nAnalysis complete. Results saved to {output_dir} directory.")

if __name__ == "__main__":
    main() 
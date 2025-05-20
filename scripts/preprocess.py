import h5py
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm

# Dictionaries to store summary statistics
data_dict = {}
length_dict = {}

def remove_extra(data_frame):
    """
    Remove MongoDB-specific extra columns if present.
    """
    for col in ['x.$numberDouble', 'y.$numberDouble', 'z.$numberDouble']:
        if col in data_frame:
            data_frame.drop(col, axis=1, inplace=True)

def json_process(directory_path):
    """
    Process all JSON files in a directory, normalize and pivot sensor data,
    and save fixed-length windows to an HDF5 file.
    """
    files = os.listdir(directory_path)
    j = 0
    total = 0

    loop = tqdm(files, total=len(files), leave=False)
    for file in loop:
        if file.endswith('.json'):
            current_directory = os.getcwd()
            file_path = os.path.join(current_directory, directory_path, file)
            df = pd.read_json(file_path)
            activity = df['activity'].values[0]
            seglength = df['elapsedTime'].values[0]

            loop.set_description(f"Processing {activity}")

            # Drop unnecessary columns
            df.drop(columns=['activity', 'elapsedTime', 'uid'], inplace=True)

            # Reset index and explode sensorData
            df = df.reset_index().rename(columns={'index': 'sensor'})
            df_exploded = df.explode('sensorData').reset_index(drop=True)

            # Normalize sensorData and merge with sensor type
            df_normalized = pd.concat([
                df_exploded['sensor'],
                pd.json_normalize(df_exploded['sensorData'])
            ], axis=1)

            # Drop timestamp column
            df_normalized.drop(columns=['t'], inplace=True)

            # Add row index for pivoting
            df_normalized['row'] = df_normalized.groupby('sensor').cumcount()

            # Pivot to wide format: each sensor type as columns
            df_pivot = df_normalized.pivot(index='row', columns='sensor', values=['x', 'y', 'z'])
            df_pivot.columns = [f'{coord}_{sensor}' for coord, sensor in df_pivot.columns]
            df_pivot = df_pivot.reset_index(drop=True)

            # Ensure consistent column order
            sorted_columns = [
                "x_accelerometer", "y_accelerometer", "z_accelerometer", 
                "x_gyroscope", "y_gyroscope", "z_gyroscope",
                "x_magnetometer", "y_magnetometer", "z_magnetometer",
                "x_absOrientation", "y_absOrientation", "z_absOrientation",
                "x_relOrientation", "y_relOrientation", "z_relOrientation"
            ]
            df_pivot = df_pivot.reindex(columns=sorted_columns)
            
            # Sliding window extraction
            num_rows = df_pivot.shape[0]
            num_splits = (num_rows - 90) // 10
            with h5py.File('../data/preprocessed/test.h5', 'a') as hf:
                for i in range(num_splits):
                    start_row = i * 10
                    if df_pivot.iloc[start_row].isnull().any():
                        continue
                    end_row = i * 10 + 100
                    df_split = df_pivot.iloc[start_row:end_row]
                    dataset_name = f'{activity}_{j}_{i}'
                    hf.create_dataset(dataset_name, data=df_split.values)
                    hf[dataset_name].attrs['activity'] = activity
                    total += 1
                    loop.set_postfix(total=total)

            # Update summary statistics
            if activity not in data_dict:
                data_dict[activity] = [num_splits, 1]
                length_dict[activity] = (num_rows * 50) / 1000
            else:
                data_dict[activity][0] += num_splits
                data_dict[activity][1] += 1
                length_dict[activity] += ((num_rows * 50) / 1000)

            j += 1

    print(length_dict)

            
            
def h5_process():
    """
    Example function for processing a specific JSON file and saving windows to HDF5.
    Not used in main execution.
    """
    json_file_path = '../data/raw/mobile-sensor-reading_acce_gyro_magnet.json'
    print(f"Reading JSON file: {json_file_path}")
    df = pd.read_json(json_file_path)
    print("JSON file read successfully.")

    # Extract sensor data columns
    df['AcceExtractedData'] = df['sensorData'].apply(lambda x: x['acce'])
    df['GyroExtractedData'] = df['sensorData'].apply(lambda x: x['gyro'])
    df['MagnetExtractedData'] = df['sensorData'].apply(lambda x: x['magnet'])

    df_stats = pd.concat([df['activity'], df['elapsedTime']], axis=1)

    num_rows = len(df_stats)
    index = 0

    while index < num_rows:
        current_row = df.iloc[index]

        # Normalize and clean accelerometer and gyroscope data
        acce_flat_data = pd.json_normalize(current_row['AcceExtractedData'])
        gyro_flat_data = pd.json_normalize(current_row['GyroExtractedData'])
        column_names = {'t': 't', 'x': 'x', 'y': 'y', 'z': 'z', '_id.$oid': 'obj_id'}
        acce_flat_data.rename(columns=column_names, inplace=True)
        gyro_flat_data.rename(columns=column_names, inplace=True)
        if 'obj_id' in acce_flat_data:
            acce_flat_data.drop('obj_id', axis=1, inplace=True)
        if 'obj_id' in gyro_flat_data:
            gyro_flat_data.drop('obj_id', axis=1, inplace=True)
        remove_extra(acce_flat_data)
        remove_extra(gyro_flat_data)
        acce_flat_data = acce_flat_data.apply(pd.to_numeric)
        gyro_flat_data = gyro_flat_data.apply(pd.to_numeric)

        # Skip empty data
        if gyro_flat_data.empty or acce_flat_data.empty:
            index += 1
            continue

        # Set time index and resample to 20ms intervals
        acce_flat_data['t'] = pd.to_datetime(acce_flat_data['t'], unit='ms')
        gyro_flat_data['t'] = pd.to_datetime(gyro_flat_data['t'], unit='ms')
        acce_flat_data.set_index('t', inplace=True)
        gyro_flat_data.set_index('t', inplace=True)
        acce_resampled = acce_flat_data.resample('20ms').mean().interpolate()
        gyro_resampled = gyro_flat_data.resample('20ms').mean().interpolate()

        # Align time indices
        start_time = min(acce_resampled.index.min(), gyro_resampled.index.min())
        end_time = max(acce_resampled.index.max(), gyro_resampled.index.max())
        common_time_index = pd.date_range(start=start_time, end=end_time, freq='20ms')
        acce_resampled = acce_resampled.reindex(common_time_index).interpolate()
        gyro_resampled = gyro_resampled.reindex(common_time_index).interpolate()

        # Sliding window extraction
        window_size = 5120
        window_start = acce_resampled.index[0]
        window_end = window_start + pd.Timedelta(milliseconds=window_size)
        window_counter = 0

        while window_end <= acce_resampled.index[-1]:
            acce_window = acce_resampled[(acce_resampled.index >= window_start) & (acce_resampled.index < window_end)]
            gyro_window = gyro_resampled[(gyro_resampled.index >= window_start) & (gyro_resampled.index < window_end)]

            if acce_window.empty or gyro_window.empty:
                window_start = window_end
                window_end = window_start + pd.Timedelta(milliseconds=window_size)
                continue

            acce_window = acce_window[['x', 'y', 'z']]
            gyro_window = gyro_window[['x', 'y', 'z']]
            sensor_window = pd.concat([acce_window, gyro_window], axis=1)
            sensor_window.columns = ['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z']

            activity = current_row['activity']
            tf_array = sensor_window[['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z']].to_numpy()
            with h5py.File('../data/preprocessed/data_xyz.h5', 'a') as hf:
                dataset = hf.create_dataset(f'data_{index}_{window_counter}', data=tf_array)
                dataset.attrs['activity'] = activity

            window_counter += 1
            window_start = window_end
            window_end = window_start + pd.Timedelta(milliseconds=window_size)
            
        index += 1
        print(f"{index}: {window_counter} windows of {activity} activity extracted.")

if __name__ == '__main__':
    # Run JSON processing on the specified directory
    json_process('../data/mstraka/api_data')
    print(data_dict)

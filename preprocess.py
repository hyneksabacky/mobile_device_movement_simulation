import h5py
import matplotlib.pyplot as plt
import pandas as pd

def remove_extra(data_frame):
    if 'x.$numberDouble' in data_frame:
        data_frame.drop('x.$numberDouble', axis=1, inplace=True)
    if 'y.$numberDouble' in data_frame:
        data_frame.drop('y.$numberDouble', axis=1, inplace=True)
    if 'z.$numberDouble' in data_frame:
        data_frame.drop('z.$numberDouble', axis=1, inplace=True)

def preprocess():
    json_file_path = 'data/raw/mobile-sensor-reading_acce_gyro_magnet.json'
    print(f"Reading JSON file: {json_file_path}")
    df = pd.read_json(json_file_path)
    print("JSON file read successfully.")

    df['AcceExtractedData'] = df['sensorData'].apply(lambda x: x['acce'])
    df['GyroExtractedData'] = df['sensorData'].apply(lambda x: x['gyro'])
    df['MagnetExtractedData'] = df['sensorData'].apply(lambda x: x['magnet'])

    df_stats = pd.concat([df['activity'], df['elapsedTime']], axis=1)
    df_stats

    num_rows = len(df_stats)
    index = 0

    while index < num_rows:
        current_row = df.iloc[index]

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

        if gyro_flat_data.empty:
            del gyro_flat_data
            index += 1
            continue
        if acce_flat_data.empty:
            del acce_flat_data
            index += 1
            continue

        acce_flat_data['t'] = pd.to_datetime(acce_flat_data['t'], unit='ms')
        gyro_flat_data['t'] = pd.to_datetime(gyro_flat_data['t'], unit='ms')
        acce_flat_data.set_index('t', inplace=True)
        gyro_flat_data.set_index('t', inplace=True)

        acce_resampled = acce_flat_data.resample('20ms').mean().interpolate()
        gyro_resampled = gyro_flat_data.resample('20ms').mean().interpolate()

        start_time = min(acce_resampled.index.min(), gyro_resampled.index.min())
        end_time = max(acce_resampled.index.max(), gyro_resampled.index.max())
        common_time_index = pd.date_range(start=start_time, end=end_time, freq='20ms')

        acce_resampled = acce_resampled.reindex(common_time_index).interpolate()
        gyro_resampled = gyro_resampled.reindex(common_time_index).interpolate()

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
            with h5py.File('data/preprocessed/data_xyz.h5', 'a') as hf:
                dataset = hf.create_dataset(f'data_{index}_{window_counter}', data=tf_array)
                dataset.attrs['activity'] = activity

            window_counter += 1
            window_start = window_end
            window_end = window_start + pd.Timedelta(milliseconds=window_size)
            
        index += 1
        print(f"{index}: {window_counter} windows of {activity} activity extracted.")

    print("Data extraction complete.")

if __name__ == '__main__':
    preprocess()
    
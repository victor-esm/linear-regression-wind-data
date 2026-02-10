import pandas as pd

def move_column(adf, col, pos=0):
    cols = adf.columns.tolist()
    cols.insert(pos, cols.pop(cols.index(col)))
    return adf[cols]

def monthly_statistics(adf):
    df_monthly = adf.copy()
    df_monthly['Month'] = df_monthly['Datetime'].dt.month
    df_monthly_statistics = df_monthly.groupby('Month').agg({
        'Vel. Vento (m/s)':'mean',
        'Raj. Vento (m/s)':'max',
        'Dir. Vento (°)':'mean',
        'Temp. Ins. (C)':'mean',
        'Temp. Max. (C)':'max',
        'Temp. Min. (C)':'min',
        'Umi. Ins. (%)': 'mean',
        'Umi. Max. (%)': 'max',
        'Umi. Min. (%)': 'min',
        'Pressao Ins. (hPa)': 'mean',
        'Pressao Max. (hPa)': 'max',
        'Pressao Min. (hPa)': 'min'
        }
    )
    return df_monthly_statistics

def clean_inmet_2025_df():
    # Read file
    data_file = r'A318_2025.csv'
    df = pd.read_csv(data_file, sep=';', decimal=',')

    # Drop unnecessary columns
    df.drop(columns=['Pto Orvalho Min. (C)', 'Pto Orvalho Ins. (C)', 'Pto Orvalho Max. (C)', 'Radiacao (KJ/m²)', 'Chuva (mm)'], inplace=True)

    # Group date and time into one single column
    df['Hora (UTC)'] = df['Hora (UTC)'].astype(str).str.zfill(4)
    df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Hora (UTC)'], format='%d/%m/%Y %H%M')
    df.drop(columns=['Data', 'Hora (UTC)'], inplace=True)

    df = move_column(df, 'Datetime', 0)
    df = move_column(df, 'Vel. Vento (m/s)', 1)
    df = move_column(df, 'Raj. Vento (m/s)', 2)
    df = move_column(df, 'Dir. Vento (m/s)', 3)
    df.rename(columns={'Dir. Vento (m/s)':'Dir. Vento (°)'}, inplace=True)
    df_clean = df.dropna(subset='Raj. Vento (m/s)')

    return df_clean

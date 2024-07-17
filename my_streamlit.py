import streamlit as st
import pandas as pd
import pymongo
import plotly.express as px
import calendar

# Koneksi ke MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["supermarket"]
collection = db["pengunjung"]

# Fetch data dari MongoDB
data = list(collection.find({}, {'_id': 0, 'gender': 1, 'days': 1, 'date': 1, 'total': 1}))
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# Fungsi untuk memisahkan gender menjadi kolom 'female' dan 'male'
def split_gender(df):
    df[['female', 'male']] = df['gender'].str.extract(r'Female: (\d+), Male: (\d+)').fillna(0).astype(int)
    return df

# Fungsi untuk membuat grafik batang jumlah pengunjung berdasarkan bulan
def create_bar_chart(df, title):
    fig = px.bar(df, x='date', y='total', title=title)
    st.plotly_chart(fig)

# Fungsi untuk membuat grafik batang total pengunjung berdasarkan bulan
def create_total_visitors_chart(df, title):
    gender_count = df[['female', 'male']].sum().reset_index().rename(columns={0: 'Count', 'index': 'Gender'})
    fig = px.bar(gender_count, x='Gender', y='Count', title=title)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    st.table(gender_count)

# Fungsi untuk membuat grafik garis jumlah pengunjung wanita per hari di bulan yang dipilih
def create_female_daily_chart(df, title):
    daily_female_count = df.groupby(df['date'].dt.date)['female'].sum().reset_index()
    fig = px.line(daily_female_count, x='date', y='female', title=title)
    st.plotly_chart(fig)

# Fungsi untuk membuat grafik garis jumlah pengunjung pria per hari di bulan yang dipilih
def create_male_daily_chart(df, title):
    daily_male_count = df.groupby(df['date'].dt.date)['male'].sum().reset_index()
    fig = px.line(daily_male_count, x='date', y='male', title=title)
    st.plotly_chart(fig)

# Fungsi untuk membuat grafik pie jumlah pengunjung per bulan di tahun 2024
def create_monthly_pie_chart(df, title):
    monthly_count = df[df['date'].dt.year == 2024].groupby(df['date'].dt.month)['total'].sum().reset_index()
    monthly_count = monthly_count.sort_values(by='total', ascending=False)
    monthly_count['month'] = monthly_count['date'].apply(lambda x: calendar.month_name[x])
    fig = px.pie(monthly_count, values='total', names='month', title=title)
    st.plotly_chart(fig)

# Fungsi untuk memfilter data berdasarkan bulan dan tahun yang dipilih user
def filter_data(selected_month, selected_year, df):
    filtered_df = df[(df['date'].dt.year == selected_year) & (df['date'].dt.month == selected_month)]
    return filtered_df

# Set up the main title
st.title('Dashboard Pengunjung Supermarket')

# Sidebar for filtering
st.sidebar.title('Filter Data')

# Pilihan filter berdasarkan bulan
selected_month = st.sidebar.selectbox('Pilih Bulan', calendar.month_name[1:], index=0, key='bulan')

# Pilihan filter berdasarkan tahun
selected_year = st.sidebar.selectbox('Pilih Tahun', range(df['date'].dt.year.min(), df['date'].dt.year.max() + 1), key='tahun')

# Pilihan jenis visualisasi
selected_visualization = st.sidebar.radio('Jenis Visualisasi', [
    'Jumlah Pengunjung', 
    'Total Pengunjung', 
    'Jumlah Female Harian', 
    'Jumlah Male Harian',
    'Urutan Pengunjung per Bulan di 2024'
])

# Filter data berdasarkan bulan dan tahun yang dipilih
filtered_df = filter_data(calendar.month_name[1:].index(selected_month) + 1, selected_year, df)

# Tampilkan visualisasi sesuai pilihan jenis visualisasi
if selected_visualization == 'Jumlah Pengunjung':
    if not filtered_df.empty:
        create_bar_chart(filtered_df, f'Jumlah Pengunjung Bulan {selected_month} {selected_year}')
    else:
        st.write(f"Tidak ada data untuk bulan {selected_month} {selected_year}.")
elif selected_visualization == 'Total Pengunjung':
    if not filtered_df.empty:
        filtered_df = split_gender(filtered_df)
        create_total_visitors_chart(filtered_df, f'Total Pengunjung Bulan {selected_month} {selected_year}')
    else:
        st.write(f"Tidak ada data untuk bulan {selected_month} {selected_year}.")
elif selected_visualization == 'Jumlah Female Harian':
    if not filtered_df.empty:
        filtered_df = split_gender(filtered_df)
        create_female_daily_chart(filtered_df, f'Jumlah Female Harian Bulan {selected_month} {selected_year}')
    else:
        st.write(f"Tidak ada data untuk bulan {selected_month} {selected_year}.")
elif selected_visualization == 'Jumlah Male Harian':
    if not filtered_df.empty:
        filtered_df = split_gender(filtered_df)
        create_male_daily_chart(filtered_df, f'Jumlah Male Harian Bulan {selected_month} {selected_year}')
    else:
        st.write(f"Tidak ada data untuk bulan {selected_month} {selected_year}.")
elif selected_visualization == 'Urutan Pengunjung per Bulan di 2024':
    create_monthly_pie_chart(df, 'Urutan Jumlah Pengunjung per Bulan di Tahun 2024')

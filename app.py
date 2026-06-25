import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Bandung FloodGuard (Sistem Prediksi Hujan dan Risiko Banjir)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD MODEL DAN DATA
@st.cache_resource
def load_model():
    """Memuat model yang telah dilatih"""
    try:
        model_package = joblib.load("flood_prediction_model.pkl")
        return model_package
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Pastikan file 'flood_prediction_model.pkl' ada di direktori yang sama.")
        return None

@st.cache_data
def load_data():
    """Memuat data historis BMKG"""
    try:
        df = pd.read_excel("datasetbmkg.xlsx", skiprows=7)
        
        # Preprocessing data
        df['RR'] = df['RR'].replace(8888, pd.NA)
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], dayfirst=True, errors='coerce')
        
        numeric_cols = ['TAVG', 'RH_AVG', 'RR', 'FF_AVG']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Rename columns
        df = df.rename(columns={
            'TAVG': 'SUHU',
            'RH_AVG': 'KELEMBAPAN',
            'RR': 'CURAH_HUJAN',
            'FF_AVG': 'KECEPATAN_ANGIN'
        })
        
        # Feature Engineering
        df['TAHUN'] = df['TANGGAL'].dt.year
        df['BULAN'] = df['TANGGAL'].dt.month
        
        # Outlier handling
        df = df[df['SUHU'] > 15].copy()
        df = df[df['KELEMBAPAN'] > 30].copy()
        Q99 = df['CURAH_HUJAN'].quantile(0.99)
        df['CURAH_HUJAN'] = df['CURAH_HUJAN'].clip(upper=Q99)
        
        return df
    except Exception as e:
        st.error(f"Error memuat data: {e}")
        return None

# FUNGSI PREDIKSI
def predict_intensity(model, features):
    """Prediksi intensitas hujan"""
    return model.predict(features)[0]

def predict_risk(intensity):
    """Konversi intensitas hujan ke tingkat risiko banjir"""
    mapping = {
        "Tidak Hujan": "Aman",
        "Ringan": "Aman",
        "Sedang": "Siaga",
        "Lebat": "Waspada"
    }
    return mapping.get(intensity, "Aman")

def get_recommendation(risk_level, intensity):
    """Rekomendasi mitigasi berdasarkan tingkat risiko"""
    recommendations = {
        "Aman": {
            "icon": "🟢",
            "title": "KONDISI AMAN",
            "message": "Tidak ada potensi banjir. Aktivitas normal dapat dilakukan.",
            "actions": [
                "Tetap waspada terhadap perubahan cuaca",
                "Pantau informasi cuaca terkini",
                "Jaga kebersihan saluran air di sekitar lingkungan"
            ]
        },
        "Siaga": {
            "icon": "🟡",
            "title": "KONDISI SIAGA",
            "message": "Potensi banjir sedang. Waspada terhadap peningkatan curah hujan.",
            "actions": [
                "Siapkan perlengkapan darurat (dokumen penting, obat-obatan)",
                "Bersihkan saluran air/selokan di sekitar rumah",
                "Hindari beraktivitas di daerah rawan banjir",
                "Pantau terus perkembangan cuaca melalui BMKG"
            ]
        },
        "Waspada": {
            "icon": "🔴",
            "title": "KONDISI WASPADA",
            "message": "Potensi banjir tinggi. Segera lakukan persiapan menghadapi banjir!",
            "actions": [
                "EVAKUASI ke tempat yang lebih tinggi jika diperlukan",
                "Amankan barang-barang berharga",
                "Matikan aliran listrik jika air mulai naik",
                "Siapkan tas darurat berisi perlengkapan penting",
                "Hubungi pihak berwenang untuk bantuan evakuasi"
            ]
        }
    }
    return recommendations.get(risk_level, recommendations["Aman"])

# FUNGSI VISUALISASI
def plot_trend_chart(df, column, title, color):
    """Membuat grafik tren variabel berdasarkan tanggal asli"""
    fig = go.Figure()
    
    df_sorted = df.sort_values('TANGGAL')
    
    fig.add_trace(go.Scatter(
        x=df_sorted['TANGGAL'],
        y=df_sorted[column],
        mode='lines+markers',
        name=column,
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color, opacity=0.7)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal",
        yaxis_title="Nilai",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        xaxis=dict(
            tickformat='%d-%b-%Y',  
            tickangle=-45,          
            nticks=20               
        )
    )
    
    return fig

def plot_correlation_heatmap(df, features):
    """Membuat heatmap korelasi"""
    corr = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdYlBu_r',
        zmin=-1,
        zmax=1,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Korelasi Antar Variabel Meteorologi",
        height=500,
        width=600,
        template='plotly_white'
    )
    
    return fig

def plot_seasonal_pattern(df, feature, title, color):
    """Membuat grafik pola musiman"""
    monthly = df.groupby('BULAN')[feature].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',
              'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months,
        y=monthly.values,
        marker_color=color,
        text=monthly.values.round(1),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Bulan",
        yaxis_title="Rata-rata",
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_rain_distribution(df):
    """Membuat distribusi curah hujan"""
    # Histogram distribusi curah hujan
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df['CURAH_HUJAN'],
        nbinsx=30,
        name='Curah Hujan',
        marker_color='#2196F3',
        opacity=0.7
    ))
    fig_hist.update_layout(
        title="Distribusi Curah Hujan Harian",
        xaxis_title="Curah Hujan (mm)",
        yaxis_title="Frekuensi",
        template='plotly_white',
        height=400
    )
    
    # Pie chart hari hujan vs tidak hujan
    rain_days = (df['CURAH_HUJAN'] > 0).sum()
    no_rain_days = (df['CURAH_HUJAN'] == 0).sum()
    
    fig_pie = go.Figure()
    fig_pie.add_trace(go.Pie(
        labels=['Hari Hujan', 'Hari Tidak Hujan'],
        values=[rain_days, no_rain_days],
        marker_colors=['#2196F3', '#FFC107'],
        hole=0.4,
        textinfo='label+percent'
    ))
    fig_pie.update_layout(
        title="Proporsi Hari Hujan vs Tidak Hujan",
        height=400,
        template='plotly_white'
    )
    
    return fig_hist, fig_pie

# FUNGSI METRIK PERFORMANCE
def display_performance_metrics():
    """Menampilkan metrik performa model"""
    metrics = {
        "Akurasi": {"value": "99.24%", "status": "Excellent"},
        "Precision (Macro Avg)": {"value": "92%", "status": "Good"},
        "Recall (Macro Avg)": {"value": "98%", "status": "Excellent"},
        "F1-Score (Macro Avg)": {"value": "94%", "status": "Good"}
    }
    
    cols = st.columns(4)
    for i, (metric, data) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(
                label=metric,
                value=data["value"],
                delta=data["status"],
                delta_color="normal"
            )
    
    st.markdown("---")
    st.markdown("""
    **Keterangan Metrik:**
    - **Akurasi**: Persentase prediksi yang benar secara keseluruhan
    - **Precision**: Ketepatan model dalam memprediksi kelas positif
    - **Recall**: Kemampuan model menemukan seluruh data positif
    - **F1-Score**: Rata-rata harmonis precision dan recall
    """)

# HALAMAN UTAMA
def main():
    # Header
    st.title("🌊 Bandung FloodGuard (Sistem Prediksi Hujan dan Risiko Banjir)")
    st.markdown("""
    <div style='background-color: #E3F2FD; padding: 1rem; border-radius: 10px; margin-bottom: 2rem'>
        <p style='margin:0; font-size:1.1rem'>
        Aplikasi ini menggunakan <b>Machine Learning (Random Forest)</b> untuk memprediksi 
        intensitas hujan dan tingkat risiko banjir berdasarkan data meteorologi (suhu, kelembapan, 
        kecepatan angin, dan curah hujan).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model dan data
    model_package = load_model()
    df = load_data()
    
    if model_package is None or df is None:
        st.stop()
    
    model = model_package["model"]
    features = model_package["features"]
    
    # Sidebar untuk input data
    with st.sidebar:
        st.markdown("## 🔍 Input Data Cuaca")
        st.markdown("Masukkan data meteorologi untuk prediksi risiko banjir:")
        
        col1, col2 = st.columns(2)
        with col1:
            suhu = st.number_input(
                "Suhu (°C)",
                min_value=15.0,
                max_value=35.0,
                value=25.0,
                step=0.1,
                help="Suhu udara dalam derajat Celsius"
            )
            kelembapan = st.number_input(
                "Kelembapan (%)",
                min_value=30.0,
                max_value=100.0,
                value=80.0,
                step=0.5,
                help="Kelembapan udara dalam persen"
            )
        with col2:
            curah_hujan = st.number_input(
                "Curah Hujan (mm)",
                min_value=0.0,
                max_value=150.0,
                value=10.0,
                step=0.5,
                help="Curah hujan dalam milimeter"
            )
            kecepatan_angin = st.number_input(
                "Kecepatan Angin (m/s)",
                min_value=0.0,
                max_value=20.0,
                value=1.0,
                step=0.5,
                help="Kecepatan angin dalam meter per detik"
            )
        
        predict_button = st.button("🔮 PREDIKSI RISIKO BANJIR", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📌 Informasi")
        st.caption(f"Model dilatih dengan data BMKG (2024-2026). Akurasi model mencapai 99.24%")
    
    # PREDIKSI
    if predict_button:
        # Buat dataframe input
        input_data = pd.DataFrame([[curah_hujan, suhu, kelembapan, kecepatan_angin]], 
                                  columns=features)
        
        # Prediksi
        prediction = predict_intensity(model, input_data)
        risk_level = predict_risk(prediction)
        recommendation = get_recommendation(risk_level, prediction)
        
        # Confidence score
        proba = model.predict_proba(input_data)
        confidence = max(proba[0]) * 100
        
        # Header hasil prediksi
        st.markdown("## 📋 HASIL PREDIKSI")
        
        # Metrik prediksi
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Intensitas Hujan", prediction)
        with col2:
            st.metric("Tingkat Risiko Banjir", risk_level)
        with col3:
            st.metric("Confidence Score", f"{confidence:.1f}%")
        
        # Rekomendasi mitigasi
        st.markdown("---")
        st.markdown(f"## {recommendation['icon']} {recommendation['title']}")
        st.info(recommendation['message'])
        
        st.markdown("### Rekomendasi Tindakan:")
        for action in recommendation['actions']:
            st.markdown(f"- {action}")
    
    # TAB VISUALISASI
    st.markdown("## 📊 VISUALISASI DATA HISTORIS")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌧️ Tren Curah Hujan", 
        "🌡️ Tren Meteorologi", 
        "📈 Korelasi",
        "🎯 Testing & Validasi"
    ])
    
    with tab1:
        st.subheader("Analisis Tren Curah Hujan Historis")
        
        col1, col2 = st.columns(2)
        with col1:
            # Tren bulanan
            fig_rain_trend = plot_trend_chart(df, 'CURAH_HUJAN', 
                                               'Tren Curah Hujan Bulanan', '#2196F3')
            st.plotly_chart(fig_rain_trend, use_container_width=True)
        
        with col2:
            # Pola musiman
            fig_rain_seasonal = plot_seasonal_pattern(df, 'CURAH_HUJAN',
                                                       'Pola Musiman Curah Hujan', '#2196F3')
            st.plotly_chart(fig_rain_seasonal, use_container_width=True)
        
        # Distribusi curah hujan (dua grafik terpisah)
        st.subheader("Distribusi Curah Hujan")
        fig_hist, fig_pie = plot_rain_distribution(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Insight
        with st.expander("📖 Insight Pola Curah Hujan"):
            st.markdown("""
            **Kesimpulan Analisis Curah Hujan:**
            - Musim hujan utama terjadi pada bulan **November - Maret** dengan puncak curah hujan
            - Periode **Juni - September** merupakan musim kemarau dengan curah hujan minimal
            - Sekitar **61%** hari dalam setahun adalah hari hujan
            - Rata-rata curah hujan harian: **~6.4 mm**
            - Intensitas hujan lebat (>50 mm) jarang terjadi (<2% dari total hari hujan)
            """)
    
    with tab2:
        st.subheader("Analisis Tren Variabel Meteorologi Pendukung")
        
        col1, col2 = st.columns(2)
        with col1:
            # Pola suhu musiman
            fig_temp_seasonal = plot_seasonal_pattern(df, 'SUHU', 
                                                       'Pola Musiman Suhu', '#FF5722')
            st.plotly_chart(fig_temp_seasonal, use_container_width=True)

            # Tren kelembapan
            fig_hum = plot_trend_chart(df, 'KELEMBAPAN', 'Tren Kelembapan Bulanan', '#4CAF50')
            st.plotly_chart(fig_hum, use_container_width=True)
            
        with col2:
            # Tren suhu
            fig_temp = plot_trend_chart(df, 'SUHU', 'Tren Suhu Bulanan', '#FF5722')
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Tren kecepatan angin
            fig_wind = plot_trend_chart(df, 'KECEPATAN_ANGIN', 'Tren Kecepatan Angin Bulanan', '#9C27B0')
            st.plotly_chart(fig_wind, use_container_width=True)
        
        # Insight
        with st.expander("📖 Insight Pola Meteorologi"):
            st.markdown("""
            **Kesimpulan Analisis Meteorologi:**
            - **Suhu**: Rata-rata 24.2°C, dengan variasi kecil sepanjang tahun (khas tropis)
            - **Kelembapan**: Rata-rata 78%, tertinggi pada bulan Desember-Januari, terendah Agustus-September
            - **Kecepatan Angin**: Umumnya rendah (0-2 m/s), mayoritas hari dengan kecepatan angin 0-1 m/s
            - Hubungan antara kelembapan tinggi dan curah hujan tinggi: **korelasi positif 0.43**
            - Suhu cenderung lebih rendah pada saat hujan (korelasi negatif -0.29)
            """)
    
    with tab3:
        st.subheader("Analisis Korelasi Antar Variabel")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # Heatmap korelasi
            features_corr = ['SUHU', 'KELEMBAPAN', 'CURAH_HUJAN', 'KECEPATAN_ANGIN']
            fig_corr = plot_correlation_heatmap(df, features_corr)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("### Korelasi dengan Curah Hujan")
            corr_data = {
                "Variabel": ["Kelembapan", "Kecepatan Angin", "Suhu"],
                "Korelasi": [0.432, -0.170, -0.287],
                "Arah": ["Positif", "Negatif", "Negatif"]
            }
            corr_df = pd.DataFrame(corr_data)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Interpretasi:**
            - 🔴**Kelembapan** memiliki pengaruh positif terkuat terhadap curah hujan
            - 🟡**Suhu** cenderung turun saat curah hujan tinggi
            - 🔵**Kecepatan angin** memiliki korelasi lemah negatif
            """)
        
        # Feature Importance
        st.markdown("### Feature Importance (Random Forest)")
        importance_data = {
            "Fitur": ["CURAH_HUJAN", "KELEMBAPAN", "SUHU", "KECEPATAN_ANGIN"],
            "Importance": [0.884, 0.085, 0.029, 0.003]
        }
        imp_df = pd.DataFrame(importance_data)
        
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=imp_df['Importance'],
            y=imp_df['Fitur'],
            orientation='h',
            marker_color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'],
            text=imp_df['Importance'].apply(lambda x: f"{x:.3f}"),
            textposition='outside'
        ))
        fig_imp.update_layout(
            title="Kontribusi Fitur terhadap Prediksi",
            xaxis_title="Importance Score",
            yaxis_title="Fitur",
            template='plotly_white',
            height=350
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with tab4:
        st.subheader("Testing dan Validasi Sistem")
        
        # Performance Metrics
        display_performance_metrics()
        
        # Confusion Matrix Info
        st.markdown("### Classification Report")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Intensitas Hujan Classification:**
            
            | Kelas | Precision | Recall | F1-Score |
            |-------|-----------|--------|----------|
            | Tidak Hujan | 1.00 | 1.00 | 1.00 |
            | Ringan | 1.00 | 1.00 | 1.00 |
            | Sedang | 1.00 | 0.92 | 0.96 |
            | Lebat | 0.67 | 1.00 | 0.80 |
            """)
        
        with col2:
            st.markdown("""
            **Risiko Banjir Classification:**
            
            | Risiko | Precision | Recall | F1-Score |
            |--------|-----------|--------|----------|
            | Aman | 1.00 | 1.00 | 1.00 |
            | Siaga | 1.00 | 0.92 | 0.96 |
            | Waspada | 0.67 | 1.00 | 0.80 |
            """)
        
        # Feature Testing
        st.markdown("### Pengujian Model dengan Skenario")
        
        test_scenarios = [
            {"Skenario": "Hari Cerah", "Curah Hujan": 0, "Kelembapan": 60, 
             "Suhu": 28, "Angin": 1, "Prediksi": "Tidak Hujan", "Risiko": "Aman"},
            {"Skenario": "Hujan Ringan", "Curah Hujan": 10, "Kelembapan": 75, 
             "Suhu": 26, "Angin": 1, "Prediksi": "Ringan", "Risiko": "Aman"},
            {"Skenario": "Hujan Sedang", "Curah Hujan": 35, "Kelembapan": 85, 
             "Suhu": 24, "Angin": 0, "Prediksi": "Sedang", "Risiko": "Siaga"},
            {"Skenario": "Hujan Lebat", "Curah Hujan": 80, "Kelembapan": 90, 
             "Suhu": 23, "Angin": 0, "Prediksi": "Lebat", "Risiko": "Waspada"}
        ]
        
        test_scenarios_df = pd.DataFrame(test_scenarios)
        st.dataframe(test_scenarios_df, use_container_width=True, hide_index=True)
    
    # FOOTER
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8rem'>"
        "Bandung FloodGuard (Sistem Prediksi Hujan dan Risiko Banjir) | "
        "Data BMKG"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
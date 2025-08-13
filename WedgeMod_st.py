import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import io

# Set page config
st.set_page_config(layout="wide", page_title="Advanced Seismic Wedge Modeling")

# Available colormaps
COLORMAPS = [
    "RdBu", "seismic", "viridis", "plasma", 
    "inferno", "magma", "cividis", "jet",
    "rainbow", "turbo", "hsv", "coolwarm"
]

# Sidebar for user inputs
st.sidebar.header("Model Parameters")

# Model source selection
model_source = st.sidebar.radio("Model Input Source", ["Manual Parameters", "Well Log CSV"])

if model_source == "Well Log CSV":
    st.sidebar.subheader("Well Log Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV with Vp, Vs, and Density", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ['Vp', 'Vs', 'Density']
            if all(col in df.columns for col in required_cols):
                # Extract values for three layers (simple approach - takes first, middle, last values)
                vp1, vp2, vp3 = df['Vp'].iloc[0], df['Vp'].iloc[len(df)//2], df['Vp'].iloc[-1]
                vs1, vs2, vs3 = df['Vs'].iloc[0], df['Vs'].iloc[len(df)//2], df['Vs'].iloc[-1]
                rho1, rho2, rho3 = df['Density'].iloc[0], df['Density'].iloc[len(df)//2], df['Density'].iloc[-1]
                
                st.sidebar.success("Well log data loaded successfully!")
            else:
                st.sidebar.error("CSV must contain columns: Vp, Vs, Density")
                model_source = "Manual Parameters"  # Fallback to manual
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {str(e)}")
            model_source = "Manual Parameters"  # Fallback to manual

# Layer properties
st.sidebar.subheader("Layer Properties")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    vp1 = st.number_input("Layer 1 Vp (m/s)", value=2500.0 if model_source == "Manual Parameters" else vp1)
    vs1 = st.number_input("Layer 1 Vs (m/s)", value=1200.0 if model_source == "Manual Parameters" else vs1)
    rho1 = st.number_input("Layer 1 Density (g/cc)", value=1.95 if model_source == "Manual Parameters" else rho1)
with col2:
    vp2 = st.number_input("Layer 2 Vp (m/s)", value=2600.0 if model_source == "Manual Parameters" else vp2)
    vs2 = st.number_input("Layer 2 Vs (m/s)", value=1300.0 if model_source == "Manual Parameters" else vs2)
    rho2 = st.number_input("Layer 2 Density (g/cc)", value=2.0 if model_source == "Manual Parameters" else rho2)
with col3:
    vp3 = st.number_input("Layer 3 Vp (m/s)", value=2550.0 if model_source == "Manual Parameters" else vp3)
    vs3 = st.number_input("Layer 3 Vs (m/s)", value=1200.0 if model_source == "Manual Parameters" else vs3)
    rho3 = st.number_input("Layer 3 Density (g/cc)", value=1.98 if model_source == "Manual Parameters" else rho3)

# Wedge parameters
st.sidebar.subheader("Wedge Geometry")
dz_min = st.sidebar.number_input("Minimum thickness (m)", value=0.0)
dz_max = st.sidebar.number_input("Maximum thickness (m)", value=60.0)
dz_step = st.sidebar.number_input("Thickness step (m)", value=1.0)

# Wavelet parameters
st.sidebar.subheader("Wavelet Parameters")
wvlt_type = st.sidebar.selectbox("Wavelet type", ["Ricker", "Bandpass"])
wvlt_length = st.sidebar.number_input("Wavelet length (s)", value=0.128)
wvlt_phase = st.sidebar.number_input("Wavelet phase (degrees)", value=0.0)
wvlt_scalar = st.sidebar.number_input("Wavelet amplitude scalar", value=1.0)

if wvlt_type == "Ricker":
    wvlt_cfreq = st.sidebar.number_input("Central frequency (Hz)", value=30.0)
else:
    f1 = st.sidebar.number_input("Low truncation freq (Hz)", value=5.0)
    f2 = st.sidebar.number_input("Low cut freq (Hz)", value=10.0)
    f3 = st.sidebar.number_input("High cut freq (Hz)", value=50.0)
    f4 = st.sidebar.number_input("High truncation freq (Hz)", value=65.0)

# Time parameters
st.sidebar.subheader("Time Parameters")
tmin = st.sidebar.number_input("Start time (s)", value=0.0)
tmax = st.sidebar.number_input("End time (s)", value=0.5)
dt = st.sidebar.number_input("Time step (s)", value=0.0001)

# Display parameters
st.sidebar.subheader("Display Parameters")
min_plot_time = st.sidebar.number_input("Min display time (s)", value=0.15)
max_plot_time = st.sidebar.number_input("Max display time (s)", value=0.3)
colormap = st.sidebar.selectbox("Seismic Colormap", COLORMAPS)
show_wiggle = st.sidebar.checkbox("Show Wiggle Traces", value=True)
wiggle_excursion = st.sidebar.number_input("Wiggle Excursion", value=0.5) if show_wiggle else 0.5
fill_positive = st.sidebar.checkbox("Fill Positive Wiggle", value=True) if show_wiggle else False
fill_color = st.sidebar.color_picker("Fill Color", "#4B8BBE") if fill_positive else None

# Main app
st.title("Advanced Seismic Wedge Modeling App")
st.write("Interactive wedge modeling with well log import and enhanced visualization")

# Functions (same as before but with enhanced wiggle fill)
def create_seismic_plot(syn_zo, t, lyr_times, min_plot_time, max_plot_time, 
                       colormap, show_wiggle, wiggle_excursion, fill_positive, fill_color):
    fig = make_subplots(rows=1, cols=1)
    
    # Heatmap
    fig.add_trace(go.Heatmap(
        z=syn_zo.T,
        x=np.arange(syn_zo.shape[0]),
        y=t,
        colorscale=colormap,
        zmin=-np.max(np.abs(syn_zo)),
        zmax=np.max(np.abs(syn_zo)),
        showscale=True,
        name='Seismic Amplitude'
    ))
    
    # Wiggle traces with optional fill
    if show_wiggle:
        ntraces = syn_zo.shape[0]
        nsamples = syn_zo.shape[1]
        normalized_data = syn_zo / np.max(np.abs(syn_zo)) * wiggle_excursion
        
        for i in range(0, ntraces, max(1, ntraces//50)):
            x_vals = np.full(nsamples, i) + normalized_data[i, :]
            
            # Main wiggle trace
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=t,
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))
            
            # Fill positive area if enabled
            if fill_positive:
                fill_x = np.where(normalized_data[i, :] > 0, x_vals, i)
                fig.add_trace(go.Scatter(
                    x=fill_x,
                    y=t,
                    fill='tozerox',
                    fillcolor=fill_color,
                    mode='none',
                    showlegend=False
                ))
    
    # Layer boundaries
    fig.add_trace(go.Scatter(
        x=np.arange(lyr_times.shape[0]),
        y=lyr_times[:, 0],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Upper Interface'
    ))
    
    fig.add_trace(go.Scatter(
        x=np.arange(lyr_times.shape[0]),
        y=lyr_times[:, 1],
        mode='lines',
        line=dict(color='red', width=2),
        name='Lower Interface'
    ))
    
    # Update layout
    fig.update_layout(
        height=700,
        title='Seismic Wedge Model',
        xaxis_title='Trace Number (Thickness)',
        yaxis_title='Time (s)',
        yaxis=dict(autorange='reversed'),
        hovermode='closest',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Set time range
    fig.update_yaxes(range=[max_plot_time, min_plot_time])
    
    return fig

# [Keep all other existing functions: ricker(), wvlt_bpass(), calc_rc(), calc_times(), digitize_model(), create_amplitude_plot()]

# Generate synthetic data
with st.spinner('Generating synthetic data...'):
    vp_mod = [vp1, vp2, vp3]
    vs_mod = [vs1, vs2, vs3]
    rho_mod = [rho1, rho2, rho3]
    nmodel = int((dz_max-dz_min)/dz_step + 1)
    
    # Generate wavelet
    if wvlt_type == 'Ricker':
        wvlt_t, wvlt_amp = ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)
    else:
        wvlt_t, wvlt_amp = wvlt_bpass(f1, f2, f3, f4, wvlt_phase, dt, wvlt_length)
    
    wvlt_amp *= wvlt_scalar
    rc_int = calc_rc(vp_mod, rho_mod)
    
    syn_zo = []
    lyr_times = []
    t = np.arange(tmin, tmax, dt)
    
    for model in range(nmodel):
        z_int = [500.0, 500.0 + dz_min + dz_step*model]
        t_int = calc_times(z_int, vp_mod)
        lyr_times.append(t_int)
        rc = digitize_model(rc_int, t_int, t)
        syn_zo.append(np.convolve(rc, wvlt_amp, mode='same'))
    
    syn_zo = np.array(syn_zo)
    lyr_times = np.array(lyr_times)
    lyr_indx = np.round(lyr_times/dt).astype(int)
    tuning_thickness = (np.argmax(np.abs(syn_zo[:, lyr_indx[0, 0]])) * dz_step + dz_min)

# Display results
tab1, tab2, tab3 = st.tabs(["Seismic Model", "Amplitude Analysis", "Wavelet"])

with tab1:
    st.subheader("Seismic Wedge Model")
    seismic_fig = create_seismic_plot(
        syn_zo, t, lyr_times, min_plot_time, max_plot_time, 
        colormap, show_wiggle, wiggle_excursion, fill_positive, fill_color
    )
    st.plotly_chart(seismic_fig, use_container_width=True)

with tab2:
    st.subheader("Amplitude Analysis")
    amplitude_fig = create_amplitude_plot(syn_zo, lyr_indx, tuning_thickness)
    st.plotly_chart(amplitude_fig, use_container_width=True)
    
    st.metric("Tuning Thickness", f"{tuning_thickness:.2f} m")

with tab3:
    st.subheader("Source Wavelet")
    wavelet_fig = go.Figure()
    wavelet_fig.add_trace(go.Scatter(
        x=wvlt_t,
        y=wvlt_amp,
        line=dict(color='blue', width=2),
        name='Wavelet'
    ))
    wavelet_fig.update_layout(
        title='Wavelet',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude'
    )
    st.plotly_chart(wavelet_fig, use_container_width=True)

st.success('Modeling complete!')

import streamlit as st
import numpy as np
import scipy.signal as signal
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

# Set page config
st.set_page_config(layout="wide", page_title="Seismic Wedge Modeling")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")

# Layer properties
st.sidebar.subheader("Layer Properties")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    vp1 = st.number_input("Layer 1 Vp (m/s)", value=2500.0)
    vs1 = st.number_input("Layer 1 Vs (m/s)", value=1200.0)
    rho1 = st.number_input("Layer 1 Density (g/cc)", value=1.95)
with col2:
    vp2 = st.number_input("Layer 2 Vp (m/s)", value=2600.0)
    vs2 = st.number_input("Layer 2 Vs (m/s)", value=1300.0)
    rho2 = st.number_input("Layer 2 Density (g/cc)", value=2.0)
with col3:
    vp3 = st.number_input("Layer 3 Vp (m/s)", value=2550.0)
    vs3 = st.number_input("Layer 3 Vs (m/s)", value=1200.0)
    rho3 = st.number_input("Layer 3 Density (g/cc)", value=1.98)

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
colormap = st.sidebar.selectbox("Seismic Colormap", ["RdBu", "seismic", "viridis"])
show_wiggle = st.sidebar.checkbox("Show Wiggle Traces", value=True)
wiggle_excursion = st.sidebar.number_input("Wiggle Excursion", value=0.5) if show_wiggle else 0.5

# Main app
st.title("Seismic Wedge Modeling App")
st.write("Interactive wedge modeling with Plotly visualizations")

# Functions
def ricker(cfreq, phase, dt, wvlt_length):
    nsamp = int(wvlt_length/dt + 1)
    t = np.linspace(-wvlt_length/2, (wvlt_length-dt)/2, int(wvlt_length/dt))
    wvlt = (1.0 - 2.0*(np.pi**2)*(cfreq**2)*(t**2)) * np.exp(-(np.pi**2)*(cfreq**2)*(t**2))
    
    if phase != 0:
        phase = phase*np.pi/180.0
        wvlth = signal.hilbert(wvlt)
        wvlth = np.imag(wvlth)
        wvlt = np.cos(phase)*wvlt - np.sin(phase)*wvlth
    
    return t, wvlt

def wvlt_bpass(f1, f2, f3, f4, phase, dt, wvlt_length):
    nsamp = int(wvlt_length/dt + 1)
    freq = fftfreq(nsamp, dt)
    freq = fftshift(freq)
    filt = np.zeros(nsamp)
    
    # Build LF ramp
    idx = np.nonzero((np.abs(freq)>=f1) & (np.abs(freq)<f2))
    M1 = 1/(f2-f1)
    b1 = -M1*f1
    filt[idx] = M1*np.abs(freq)[idx]+b1
    
    # Build central filter flat
    idx = np.nonzero((np.abs(freq)>=f2) & (np.abs(freq)<=f3))
    filt[idx] = 1.0
    
    # Build HF ramp
    idx = np.nonzero((np.abs(freq)>f3) & (np.abs(freq)<=f4))
    M2 = -1/(f4-f3)
    b2 = -M2*f4
    filt[idx] = M2*np.abs(freq)[idx]+b2
    
    filt2 = ifftshift(filt)
    Af = filt2*np.exp(np.zeros(filt2.shape)*1j)
    wvlt = fftshift(ifft(Af))
    wvlt = np.real(wvlt)
    wvlt = wvlt/np.max(np.abs(wvlt))
    
    t = np.linspace(-wvlt_length*0.5, wvlt_length*0.5, nsamp)
    
    if phase != 0:
        phase = phase*np.pi/180.0
        wvlth = signal.hilbert(wvlt)
        wvlth = np.imag(wvlth)
        wvlt = np.cos(phase)*wvlt - np.sin(phase)*wvlth
    
    return t, wvlt

def calc_rc(vp_mod, rho_mod):
    return [(vp_mod[i+1]*rho_mod[i+1]-vp_mod[i]*rho_mod[i])/(vp_mod[i+1]*rho_mod[i+1]+vp_mod[i]*rho_mod[i]) 
            for i in range(len(vp_mod)-1)]

def calc_times(z_int, vp_mod):
    t_int = [z_int[0]/vp_mod[0]]
    for i in range(1, len(z_int)):
        t_int.append(2*(z_int[i]-z_int[i-1])/vp_mod[i] + t_int[i-1])
    return t_int

def digitize_model(rc_int, t_int, t):
    rc = np.zeros_like(t)
    lyr = 0
    for i in range(len(t)):
        if lyr < len(t_int) and t[i] >= t_int[lyr]:
            rc[i] = rc_int[lyr]
            lyr += 1
    return rc

def create_seismic_plot(syn_zo, t, lyr_times, min_plot_time, max_plot_time, colormap, show_wiggle, wiggle_excursion):
    fig = make_subplots(rows=1, cols=1)
    
    # Heatmap
    fig.add_trace(go.Heatmap(
        z=syn_zo.T,
        x=np.arange(syn_zo.shape[0]),
        y=t,
        colorscale=colormap,
        zmin=-np.max(np.abs(syn_zo)),
        zmax=np.max(np.abs(syn_zo))
    ))
    
    # Wiggle traces
    if show_wiggle:
        ntraces = syn_zo.shape[0]
        norm_data = syn_zo / np.max(np.abs(syn_zo)) * wiggle_excursion
        for i in range(0, ntraces, max(1, ntraces//50)):
            fig.add_trace(go.Scatter(
                x=np.full(len(t), i) + norm_data[i, :],
                y=t,
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))
    
    # Layer boundaries
    fig.add_trace(go.Scatter(
        x=np.arange(lyr_times.shape[0]),
        y=lyr_times[:, 0],
        line=dict(color='blue', width=2),
        name='Upper Interface'
    ))
    
    fig.add_trace(go.Scatter(
        x=np.arange(lyr_times.shape[0]),
        y=lyr_times[:, 1],
        line=dict(color='red', width=2),
        name='Lower Interface'
    ))
    
    fig.update_layout(
        height=600,
        title='Seismic Wedge Model',
        xaxis_title='Trace Number (Thickness)',
        yaxis_title='Time (s)',
        yaxis=dict(autorange='reversed'),
        hovermode='closest'
    )
    fig.update_yaxes(range=[max_plot_time, min_plot_time])
    
    return fig

def create_amplitude_plot(syn_zo, lyr_indx, tuning_thickness):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.arange(syn_zo.shape[0]),
        y=syn_zo[:, lyr_indx[0, 0]],
        line=dict(color='blue', width=2),
        name='Upper Interface Amplitude'
    ))
    
    fig.add_vline(
        x=tuning_thickness/dz_step,
        line=dict(color='black', width=2, dash='dash'),
        annotation_text=f'Tuning: {tuning_thickness:.1f} m'
    )
    
    fig.update_layout(
        height=300,
        title='Amplitude vs Thickness',
        xaxis_title='Thickness (m)',
        yaxis_title='Amplitude'
    )
    
    return fig

# Generate synthetic data
with st.spinner('Generating synthetic data...'):
    vp_mod = [vp1, vp2, vp3]
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
    tuning_thickness = (np.argmax(np.abs(syn_zo[:, lyr_indx[0, 0]])) * dz_step + dz_min

# Display results
st.subheader("Seismic Wedge Model")
seismic_fig = create_seismic_plot(syn_zo, t, lyr_times, min_plot_time, max_plot_time, 
                                 colormap, show_wiggle, wiggle_excursion)
st.plotly_chart(seismic_fig, use_container_width=True)

st.subheader("Amplitude Analysis")
amplitude_fig = create_amplitude_plot(syn_zo, lyr_indx, tuning_thickness)
st.plotly_chart(amplitude_fig, use_container_width=True)

st.subheader("Source Wavelet")
wavelet_fig = go.Figure()
wavelet_fig.add_trace(go.Scatter(
    x=wvlt_t,
    y=wvlt_amp,
    line=dict(color='blue', width=2)
))
wavelet_fig.update_layout(
    title='Wavelet',
    xaxis_title='Time (s)',
    yaxis_title='Amplitude'
)
st.plotly_chart(wavelet_fig, use_container_width=True)

st.success('Modeling complete!')

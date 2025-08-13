# seismic_wedge_modeling_app.py
import streamlit as st
import numpy as np
import scipy.signal as signal
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from scipy.interpolate import interp1d

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Seismic Wedge Modeling")

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
wvlt_type = st.sidebar.selectbox("Wavelet type", ["Ricker", "Bandpass", "PyWavelet"])
wvlt_length = st.sidebar.number_input("Wavelet length (s)", value=0.128)
wvlt_phase = st.sidebar.number_input("Wavelet phase (degrees)", value=0.0)
wvlt_scalar = st.sidebar.number_input("Wavelet amplitude scalar", value=1.0)

if wvlt_type == "Ricker":
    wvlt_cfreq = st.sidebar.number_input("Central frequency (Hz)", value=30.0)
elif wvlt_type == "Bandpass":
    f1 = st.sidebar.number_input("Low truncation freq (Hz)", value=5.0)
    f2 = st.sidebar.number_input("Low cut freq (Hz)", value=10.0)
    f3 = st.sidebar.number_input("High cut freq (Hz)", value=50.0)
    f4 = st.sidebar.number_input("High truncation freq (Hz)", value=65.0)
else:  # PyWavelet
    wavelet_families = pywt.families()
    selected_family = st.sidebar.selectbox("Wavelet Family", wavelet_families)
    wavelets = pywt.wavelist(family=selected_family)
    selected_wavelet = st.sidebar.selectbox("Wavelet Type", wavelets)
    wavelet_scale = st.sidebar.number_input("Wavelet Scale", value=1.0, min_value=0.1, max_value=10.0)

# Time parameters
st.sidebar.subheader("Time Parameters")
tmin = st.sidebar.number_input("Start time (s)", value=0.0)
tmax = st.sidebar.number_input("End time (s)", value=0.5)
dt = st.sidebar.number_input("Time step (s)", value=0.0001)

# Display parameters
st.sidebar.subheader("Display Parameters")
min_plot_time = st.sidebar.number_input("Min display time (s)", value=0.15)
max_plot_time = st.sidebar.number_input("Max display time (s)", value=0.3)
colormap = st.sidebar.selectbox("Seismic Colormap", 
                               ["RdBu", "seismic", "viridis", "plasma", "inferno", "magma", "cividis"])
show_wiggle = st.sidebar.checkbox("Show Wiggle Traces", value=True)
wiggle_excursion = st.sidebar.number_input("Wiggle Excursion", value=0.5) if show_wiggle else 0.5

# Main app
st.title("Enhanced Seismic Wedge Modeling App")
st.write("""
This app generates a synthetic seismic section for a 3-layer wedge model with interactive visualization.
Adjust parameters in the sidebar and view the results below.
""")

# Organize model parameters
vp_mod = [vp1, vp2, vp3]
vs_mod = [vs1, vs2, vs3]
rho_mod = [rho1, rho2, rho3]

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

def pywavelet_wavelet(wavelet_name, scale, phase, dt, wvlt_length):
    nsamp = int(wvlt_length/dt + 1)
    t = np.linspace(-wvlt_length/2, wvlt_length/2, nsamp)
    
    # Generate wavelet
    wavelet = pywt.ContinuousWavelet(wavelet_name)
    psi, x = pywt.ContinuousWavelet.wavefun(wavelet, level=10)
    
    # Interpolate to desired time points
    f = interp1d(x, psi, kind='cubic', fill_value='extrapolate')
    wvlt = f(t/scale)
    
    # Normalize
    wvlt = wvlt / np.max(np.abs(wvlt))
    
    if phase != 0:
        phase = phase*np.pi/180.0
        wvlth = signal.hilbert(wvlt)
        wvlth = np.imag(wvlth)
        wvlt = np.cos(phase)*wvlt - np.sin(phase)*wvlth
    
    return t, wvlt

def calc_rc(vp_mod, rho_mod):
    nlayers = len(vp_mod)
    nint = nlayers - 1
    rc_int = []
    for i in range(0, nint):
        buf1 = vp_mod[i+1]*rho_mod[i+1]-vp_mod[i]*rho_mod[i]
        buf2 = vp_mod[i+1]*rho_mod[i+1]+vp_mod[i]*rho_mod[i]
        buf3 = buf1/buf2
        rc_int.append(buf3)
    return rc_int

def calc_times(z_int, vp_mod):
    nlayers = len(vp_mod)
    nint = nlayers - 1
    t_int = []
    for i in range(0, nint):
        if i == 0:
            tbuf = z_int[i]/vp_mod[i]
            t_int.append(tbuf)
        else:
            zdiff = z_int[i]-z_int[i-1]
            tbuf = 2*zdiff/vp_mod[i] + t_int[i-1]
            t_int.append(tbuf)
    return t_int

def digitize_model(rc_int, t_int, t):
    nlayers = len(rc_int)
    nint = nlayers - 1
    nsamp = len(t)
    rc = list(np.zeros(nsamp,dtype='float'))
    lyr = 0
    
    for i in range(0, nsamp):
        if t[i] >= t_int[lyr]:
            rc[i] = rc_int[lyr]
            lyr = lyr + 1    
        if lyr > nint:
            break
    return rc

def create_seismic_plot(syn_zo, t, lyr_times, min_plot_time, max_plot_time, colormap, show_wiggle, wiggle_excursion):
    fig = make_subplots(rows=1, cols=1)
    
    # Create heatmap for seismic data
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
    
    # Add wiggle traces if enabled
    if show_wiggle:
        ntraces = syn_zo.shape[0]
        nsamples = syn_zo.shape[1]
        normalized_data = syn_zo / np.max(np.abs(syn_zo)) * wiggle_excursion
        
        for i in range(0, ntraces, max(1, ntraces//50)):  # Plot every Nth trace to avoid overcrowding
            fig.add_trace(go.Scatter(
                x=np.full(nsamples, i) + normalized_data[i, :],
                y=t,
                mode='lines',
                line=dict(color='black', width=1),
                name=f'Trace {i}',
                showlegend=False
            ))
    
    # Add layer boundaries
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
        height=600,
        title='Seismic Wedge Model',
        xaxis_title='Trace Number (Thickness)',
        yaxis_title='Time (s)',
        yaxis=dict(autorange='reversed'),
        hovermode='closest'
    )
    
    # Set time range
    fig.update_yaxes(range=[max_plot_time, min_plot_time])
    
    return fig

def create_amplitude_plot(syn_zo, lyr_indx, tuning_trace, tuning_thickness):
    fig = go.Figure()
    
    # Plot amplitude at upper interface
    fig.add_trace(go.Scatter(
        x=np.arange(syn_zo.shape[0]),
        y=syn_zo[:, lyr_indx[:, 0][0]],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Upper Interface Amplitude'
    ))
    
    # Add tuning thickness marker
    fig.add_vline(
        x=tuning_trace,
        line=dict(color='black', width=2, dash='dash'),
        annotation_text=f'Tuning Thickness: {tuning_thickness:.1f} m',
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        height=300,
        title='Upper Interface Amplitude vs Wedge Thickness',
        xaxis_title='Thickness (m)',
        yaxis_title='Amplitude',
        showlegend=True
    )
    
    return fig

# Generate synthetic data
with st.spinner('Generating synthetic data...'):
    nlayers = len(vp_mod)
    nint = nlayers - 1
    nmodel = int((dz_max-dz_min)/dz_step+1)

    # Generate wavelet
    if wvlt_type == 'Ricker':
        wvlt_t, wvlt_amp = ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)
    elif wvlt_type == 'Bandpass':
        wvlt_t, wvlt_amp = wvlt_bpass(f1, f2, f3, f4, wvlt_phase, dt, wvlt_length)
    else:
        wvlt_t, wvlt_amp = pywavelet_wavelet(selected_wavelet, wavelet_scale, wvlt_phase, dt, wvlt_length)

    wvlt_amp = wvlt_scalar * wvlt_amp
    rc_int = calc_rc(vp_mod, rho_mod)

    syn_zo = []
    rc_zo = []
    lyr_times = []
    for model in range(0, nmodel):
        z_int = [500.0]
        z_int.append(z_int[0]+dz_min+dz_step*model)
        t_int = calc_times(z_int, vp_mod)
        lyr_times.append(t_int)
        
        nsamp = int((tmax-tmin)/dt) + 1
        t = []
        for i in range(0,nsamp):
            t.append(i*dt)
            
        rc = digitize_model(rc_int, t_int, t)
        rc_zo.append(rc)
        syn_buf = np.convolve(rc, wvlt_amp, mode='same')
        syn_buf = list(syn_buf)
        syn_zo.append(syn_buf)
    
    syn_zo = np.array(syn_zo)
    t = np.array(t)
    lyr_times = np.array(lyr_times)
    lyr_indx = np.array(np.round(lyr_times/dt), dtype='int16')
    tuning_trace = np.argmax(np.abs(syn_zo.T)) % syn_zo.T.shape[1]
    tuning_thickness = tuning_trace * dz_step + dz_min

# Create and display plots
st.subheader("Seismic Wedge Model Visualization")
seismic_fig = create_seismic_plot(syn_zo, t, lyr_times, min_plot_time, max_plot_time, 
                                 colormap, show_wiggle, wiggle_excursion)
st.plotly_chart(seismic_fig, use_container_width=True)

st.subheader("Amplitude Analysis")
amplitude_fig = create_amplitude_plot(syn_zo, lyr_indx, tuning_trace, tuning_thickness)
st.plotly_chart(amplitude_fig, use_container_width=True)

# Display wavelet
st.subheader("Wavelet Visualization")
wavelet_fig = go.Figure()
wavelet_fig.add_trace(go.Scatter(
    x=wvlt_t,
    y=wvlt_amp,
    mode='lines',
    line=dict(color='blue', width=2),
    name='Wavelet'
))
wavelet_fig.update_layout(
    title='Source Wavelet',
    xaxis_title='Time (s)',
    yaxis_title='Amplitude'
)
st.plotly_chart(wavelet_fig, use_container_width=True)

st.success('Analysis complete!')

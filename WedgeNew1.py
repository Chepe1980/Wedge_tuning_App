import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================
# FUNCTION DEFINITIONS
# ==============================================

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

def create_seismic_plot(syn_zo, t, lyr_times, min_plot_time, max_plot_time, 
                       colormap, show_wiggle, wiggle_excursion, fill_positive, fill_color,
                       vp_mod, vs_mod, rho_mod):
    fig = make_subplots(rows=1, cols=1)
    
    # Heatmap background
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
    
    # Wiggle traces with proper positive fill
    if show_wiggle:
        ntraces = syn_zo.shape[0]
        nsamples = syn_zo.shape[1]
        normalized_data = syn_zo / np.max(np.abs(syn_zo)) * wiggle_excursion
        
        for i in range(0, ntraces, max(1, ntraces//50)):
            trace_x = np.full(nsamples, i)
            trace_y = t
            trace_vals = normalized_data[i, :]
            
            # Main wiggle trace
            fig.add_trace(go.Scatter(
                x=trace_x + trace_vals,
                y=trace_y,
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))
            
            # Fill only positive parts of the signal
            if fill_positive:
                # Find segments where signal is positive
                positive_segments = []
                current_segment = []
                for j in range(len(trace_vals)):
                    if trace_vals[j] > 0:
                        current_segment.append(j)
                    elif current_segment:
                        positive_segments.append(current_segment)
                        current_segment = []
                
                if current_segment:
                    positive_segments.append(current_segment)
                
                # Add fill for each positive segment
                for segment in positive_segments:
                    if len(segment) > 1:  # Need at least 2 points to fill
                        seg_x = trace_x[segment] + trace_vals[segment]
                        seg_y = trace_y[segment]
                        
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([seg_x, seg_x[::-1]]),
                            y=np.concatenate([seg_y, seg_y[::-1]]),
                            fill='toself',
                            fillcolor=fill_color,
                            mode='none',
                            showlegend=False,
                            hoverinfo='skip'
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
    
    # Add layer property annotations
    mid_trace = syn_zo.shape[0] // 2
    for i in range(len(vp_mod)):
        if i == 0:
            # For first layer, show properties above the interface
            y_pos = lyr_times[mid_trace, 0] - 0.01
            text = f"Layer {i+1}<br>Vp: {vp_mod[i]:.0f} m/s<br>Vs: {vs_mod[i]:.0f} m/s<br>ρ: {rho_mod[i]:.2f} g/cc"
        elif i == len(vp_mod)-1:
            # For last layer, show properties below the interface
            y_pos = lyr_times[mid_trace, -1] + 0.01
            text = f"Layer {i+1}<br>Vp: {vp_mod[i]:.0f} m/s<br>Vs: {vs_mod[i]:.0f} m/s<br>ρ: {rho_mod[i]:.2f} g/cc"
        else:
            # For middle layers, show properties between interfaces
            y_pos = (lyr_times[mid_trace, i-1] + lyr_times[mid_trace, i]) / 2
            text = f"Layer {i+1}<br>Vp: {vp_mod[i]:.0f} m/s<br>Vs: {vs_mod[i]:.0f} m/s<br>ρ: {rho_mod[i]:.2f} g/cc"
        
        fig.add_annotation(
            x=mid_trace,
            y=y_pos,
            text=text,
            showarrow=False,
            font=dict(size=10, color="white"),
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            bgcolor="rgba(0,0,0,0.5)"
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        title='Seismic Wedge Model (Time Domain)',
        xaxis_title='Trace Number (Thickness)',
        yaxis_title='Time (s)',
        yaxis=dict(autorange='reversed'),
        hovermode='closest',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Set time range
    fig.update_yaxes(range=[max_plot_time, min_plot_time])
    
    return fig

def create_frequency_plot(syn_zo, t, dt, min_freq, max_freq):
    # Compute frequency spectrum for each trace
    ntraces = syn_zo.shape[0]
    nsamples = syn_zo.shape[1]
    freq = fftshift(fftfreq(nsamples, dt))
    freq_mask = (freq >= min_freq) & (freq <= max_freq)
    
    # Calculate amplitude spectrum for each trace
    spec = np.zeros((ntraces, len(freq[freq_mask])))
    for i in range(ntraces):
        trace_fft = fftshift(fft(syn_zo[i,:]))
        spec[i,:] = np.abs(trace_fft[freq_mask])
    
    # Normalize the spectrum
    spec = spec / np.max(spec)
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=spec.T,
        x=np.arange(ntraces),
        y=freq[freq_mask],
        colorscale='viridis',
        zmin=0,
        zmax=1,
        showscale=True,
        name='Amplitude Spectrum'
    ))
    
    # Update layout
    fig.update_layout(
        height=700,
        title='Seismic Wedge Model (Frequency Domain)',
        xaxis_title='Trace Number (Thickness)',
        yaxis_title='Frequency (Hz)',
        hovermode='closest',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_amplitude_plot(syn_zo, lyr_indx, tuning_thickness, dz_min, dz_step):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.arange(syn_zo.shape[0]) * dz_step + dz_min,
        y=syn_zo[:, lyr_indx[0, 0]],
        line=dict(color='blue', width=2),
        name='Upper Interface Amplitude'
    ))
    
    fig.add_vline(
        x=tuning_thickness,
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

def create_well_log_plot(depths, vp_values, layer_ranges):
    fig = go.Figure()
    
    # Plot Vp curve
    fig.add_trace(go.Scatter(
        x=vp_values,
        y=depths,
        mode='lines',
        name='Vp',
        line=dict(color='blue', width=2)
    ))
    
    # Add layer ranges
    colors = ['red', 'green', 'purple']
    for i, (top, base) in enumerate(layer_ranges):
        # Highlight layer area
        fig.add_vrect(
            x0=min(vp_values), x1=max(vp_values),
            y0=top, y1=base,
            fillcolor=colors[i], opacity=0.2,
            layer="below", line_width=0,
            name=f'Layer {i+1} Range'
        )
        
        # Add boundary lines
        fig.add_hline(
            y=top,
            line=dict(color=colors[i], width=2, dash='dash'),
            annotation_text=f'Layer {i+1} Top',
            annotation_position="top right"
        )
        fig.add_hline(
            y=base,
            line=dict(color=colors[i], width=2, dash='dash'),
            annotation_text=f'Layer {i+1} Base',
            annotation_position="bottom right"
        )
    
    fig.update_layout(
        title='Well Log Visualization with Layer Ranges',
        xaxis_title='Vp (m/s)',
        yaxis_title='Depth (m)',
        yaxis=dict(autorange='reversed'),
        height=500,
        showlegend=True
    )
    
    return fig

# ==============================================
# STREAMLIT APP UI
# ==============================================

st.set_page_config(layout="wide", page_title="Advanced Seismic Wedge Modeling")

COLORMAPS = [
    "RdBu", "seismic", "viridis", "plasma", 
    "inferno", "magma", "cividis", "jet",
    "rainbow", "turbo", "hsv", "coolwarm"
]

st.sidebar.header("Model Parameters")

# Initialize default values
default_vp = [2500.0, 2600.0, 2550.0]
default_vs = [1200.0, 1300.0, 1200.0]
default_rho = [1.95, 2.0, 1.98]
default_depths = [0, 100, 200, 300]  # Top of layer 1, layer 2, layer 3, and bottom

# Well Log CSV Upload
uploaded_file = st.sidebar.file_uploader("Upload Well Log CSV (columns: Depth, Vp, Vs, Density)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in ['Depth', 'Vp', 'Vs', 'Density']):
            df = df.sort_values('Depth')
            depths = df['Depth'].values
            vp_values = df['Vp'].values
            
            st.sidebar.subheader("Layer Range Selection")
            
            # Layer range selection
            layer_ranges = []
            min_depth = depths.min()
            max_depth = depths.max()
            
            st.sidebar.write("### Layer 1 Range")
            l1_top = st.sidebar.number_input("Layer 1 Top (m)", min_value=min_depth, max_value=max_depth, value=min_depth)
            l1_base = st.sidebar.number_input("Layer 1 Base (m)", min_value=l1_top, max_value=max_depth, value=(min_depth + (max_depth-min_depth)/3))
            layer_ranges.append((l1_top, l1_base))
            
            st.sidebar.write("### Layer 2 Range")
            l2_top = st.sidebar.number_input("Layer 2 Top (m)", min_value=l1_base, max_value=max_depth, value=l1_base)
            l2_base = st.sidebar.number_input("Layer 2 Base (m)", min_value=l2_top, max_value=max_depth, value=(min_depth + 2*(max_depth-min_depth)/3))
            layer_ranges.append((l2_top, l2_base))
            
            st.sidebar.write("### Layer 3 Range")
            l3_top = st.sidebar.number_input("Layer 3 Top (m)", min_value=l2_base, max_value=max_depth, value=l2_base)
            l3_base = st.sidebar.number_input("Layer 3 Base (m)", min_value=l3_top, max_value=max_depth, value=max_depth)
            layer_ranges.append((l3_top, l3_base))
            
            # Calculate average properties for each layer
            def get_layer_avg(df, top, base, prop):
                layer_df = df[(df['Depth'] >= top) & (df['Depth'] <= base)]
                return layer_df[prop].mean()
            
            vp1 = get_layer_avg(df, l1_top, l1_base, 'Vp')
            vs1 = get_layer_avg(df, l1_top, l1_base, 'Vs')
            rho1 = get_layer_avg(df, l1_top, l1_base, 'Density')
            
            vp2 = get_layer_avg(df, l2_top, l2_base, 'Vp')
            vs2 = get_layer_avg(df, l2_top, l2_base, 'Vs')
            rho2 = get_layer_avg(df, l2_top, l2_base, 'Density')
            
            vp3 = get_layer_avg(df, l3_top, l3_base, 'Vp')
            vs3 = get_layer_avg(df, l3_top, l3_base, 'Vs')
            rho3 = get_layer_avg(df, l3_top, l3_base, 'Density')
            
            st.sidebar.success("Well log data loaded successfully!")
            
            # Show well log visualization
            st.subheader("Well Log Visualization with Selected Layer Ranges")
            well_log_fig = create_well_log_plot(depths, vp_values, layer_ranges)
            st.plotly_chart(well_log_fig, use_container_width=True)
            
        else:
            st.sidebar.error("CSV must contain columns: Depth, Vp, Vs, Density")
            raise ValueError("Missing required columns")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {str(e)}")
        # Fall back to default values
        depths = np.linspace(0, 300, 100)
        vp_values = np.piecewise(depths, 
                                [depths < 100, (depths >= 100) & (depths < 200), depths >= 200],
                                [default_vp[0], default_vp[1], default_vp[2]])
        layer_ranges = [(0, 100), (100, 200), (200, 300)]
        vp1, vs1, rho1 = default_vp[0], default_vs[0], default_rho[0]
        vp2, vs2, rho2 = default_vp[1], default_vs[1], default_rho[1]
        vp3, vs3, rho3 = default_vp[2], default_vs[2], default_rho[2]
else:
    # Manual input mode
    st.sidebar.subheader("Layer Properties")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        vp1 = st.number_input("Layer 1 Vp (m/s)", value=default_vp[0])
        vs1 = st.number_input("Layer 1 Vs (m/s)", value=default_vs[0])
        rho1 = st.number_input("Layer 1 Density (g/cc)", value=default_rho[0])
    with col2:
        vp2 = st.number_input("Layer 2 Vp (m/s)", value=default_vp[1])
        vs2 = st.number_input("Layer 2 Vs (m/s)", value=default_vs[1])
        rho2 = st.number_input("Layer 2 Density (g/cc)", value=default_rho[1])
    with col3:
        vp3 = st.number_input("Layer 3 Vp (m/s)", value=default_vp[2])
        vs3 = st.number_input("Layer 3 Vs (m/s)", value=default_vs[2])
        rho3 = st.number_input("Layer 3 Density (g/cc)", value=default_rho[2])
    
    # Create synthetic well log for visualization
    depths = np.linspace(0, 300, 100)
    vp_values = np.piecewise(depths, 
                            [depths < 100, (depths >= 100) & (depths < 200), depths >= 200],
                            [vp1, vp2, vp3])
    layer_ranges = [(0, 100), (100, 200), (200, 300)]
    
    # Show well log visualization
    st.subheader("Well Log Visualization (Synthetic)")
    well_log_fig = create_well_log_plot(depths, vp_values, layer_ranges)
    st.plotly_chart(well_log_fig, use_container_width=True)

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

# Frequency domain parameters
st.sidebar.subheader("Frequency Domain Parameters")
min_freq = st.sidebar.number_input("Min frequency (Hz)", value=0.0)
max_freq = st.sidebar.number_input("Max frequency (Hz)", value=100.0)

# Main app
st.title("Advanced Seismic Wedge Modeling App")
st.write("Interactive wedge modeling with well log visualization")

# ==============================================
# MODEL PROCESSING
# ==============================================

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

# ==============================================
# RESULTS DISPLAY
# ==============================================

tab1, tab2, tab3, tab4 = st.tabs(["Time Domain", "Frequency Domain", "Amplitude Analysis", "Wavelet"])

with tab1:
    st.subheader("Seismic Wedge Model (Time Domain)")
    seismic_fig = create_seismic_plot(
        syn_zo, t, lyr_times, min_plot_time, max_plot_time, 
        colormap, show_wiggle, wiggle_excursion, fill_positive, fill_color,
        vp_mod, vs_mod, rho_mod
    )
    st.plotly_chart(seismic_fig, use_container_width=True)

with tab2:
    st.subheader("Seismic Wedge Model (Frequency Domain)")
    freq_fig = create_frequency_plot(syn_zo, t, dt, min_freq, max_freq)
    st.plotly_chart(freq_fig, use_container_width=True)

with tab3:
    st.subheader("Amplitude Analysis")
    amplitude_fig = create_amplitude_plot(syn_zo, lyr_indx, tuning_thickness, dz_min, dz_step)
    st.plotly_chart(amplitude_fig, use_container_width=True)
    
    st.metric("Tuning Thickness", f"{tuning_thickness:.2f} m")

with tab4:
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

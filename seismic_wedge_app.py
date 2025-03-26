import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.signal as signal
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift

st.set_page_config(layout="wide", page_title="Seismic Wedge Modeling")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")

# Layer properties
st.sidebar.subheader("Layer Properties")
vp1 = st.sidebar.number_input("Layer 1 Vp (m/s)", value=2500.0)
vp2 = st.sidebar.number_input("Layer 2 Vp (m/s)", value=2600.0)
vp3 = st.sidebar.number_input("Layer 3 Vp (m/s)", value=2550.0)

vs1 = st.sidebar.number_input("Layer 1 Vs (m/s)", value=1200.0)
vs2 = st.sidebar.number_input("Layer 2 Vs (m/s)", value=1300.0)
vs3 = st.sidebar.number_input("Layer 3 Vs (m/s)", value=1200.0)

rho1 = st.sidebar.number_input("Layer 1 Density (g/cc)", value=1.95)
rho2 = st.sidebar.number_input("Layer 2 Density (g/cc)", value=2.0)
rho3 = st.sidebar.number_input("Layer 3 Density (g/cc)", value=1.98)

# Wedge parameters
st.sidebar.subheader("Wedge Geometry")
dz_min = st.sidebar.number_input("Minimum thickness (m)", value=0.0)
dz_max = st.sidebar.number_input("Maximum thickness (m)", value=60.0)
dz_step = st.sidebar.number_input("Thickness step (m)", value=1.0)

# Wavelet parameters
st.sidebar.subheader("Wavelet Parameters")
wvlt_type = st.sidebar.selectbox("Wavelet type", ["ricker", "bandpass"])
wvlt_length = st.sidebar.number_input("Wavelet length (s)", value=0.128)
wvlt_phase = st.sidebar.number_input("Wavelet phase (degrees)", value=0.0)
wvlt_scalar = st.sidebar.number_input("Wavelet amplitude scalar", value=1.0)

if wvlt_type == "ricker":
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
excursion = st.sidebar.number_input("Trace excursion", value=2)

# Main app
st.title("Seismic Wedge Modeling App")
st.write("""
This app generates a synthetic seismic section for a 3-layer wedge model.
Adjust parameters in the sidebar and view the results below.
""")

# Organize model parameters
vp_mod = [vp1, vp2, vp3]
vs_mod = [vs1, vs2, vs3]
rho_mod = [rho1, rho2, rho3]

# Functions (same as original)
def plot_vawig(axhdl, data, t, excursion, highlight=None):
    [ntrc, nsamp] = data.shape
    t = np.hstack([0, t, t.max()])
    
    for i in range(0, ntrc):
        tbuf = excursion * data[i] / np.max(np.abs(data)) + i
        tbuf = np.hstack([i, tbuf, i])
            
        if i==highlight:
            lw = 2
        else:
            lw = 0.5

        axhdl.plot(tbuf, t, color='black', linewidth=lw)
        plt.fill_betweenx(t, tbuf, i, where=tbuf>i, facecolor=[0.6,0.6,1.0], linewidth=0)
        plt.fill_betweenx(t, tbuf, i, where=tbuf<i, facecolor=[1.0,0.7,0.7], linewidth=0)
    
    axhdl.set_xlim((-excursion, ntrc+excursion))
    axhdl.xaxis.tick_top()
    axhdl.xaxis.set_label_position('top')
    axhdl.invert_yaxis()

def ricker(cfreq, phase, dt, wvlt_length):
    nsamp = int(wvlt_length/dt + 1)
    #t = np.linspace(-wvlt_length/2, (wvlt_length-dt)/2, wvlt_length/dt)
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

# Generate synthetic data
with st.spinner('Generating synthetic data...'):
    nlayers = len(vp_mod)
    nint = nlayers - 1
    nmodel = int((dz_max-dz_min)/dz_step+1)

    # Generate wavelet
    if wvlt_type == 'ricker':
        wvlt_t, wvlt_amp = ricker(wvlt_cfreq, wvlt_phase, dt, wvlt_length)
    elif wvlt_type == 'bandpass':
        wvlt_t, wvlt_amp = wvlt_bpass(f1, f2, f3, f4, wvlt_phase, dt, wvlt_length)

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
    tuning_thickness = tuning_trace * dz_step

# Create plots
fig = plt.figure(figsize=(12, 14))
fig.set_facecolor('white')
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

excursion = 10  # Set this value as per your requirements
ntrc = 100      # Set this value as per your requirements


ax0 = fig.add_subplot(gs[0])
ax0.plot(lyr_times[:,0], color='blue', lw=1.5)
ax0.plot(lyr_times[:,1], color='red', lw=1.5)
ax0.set_ylim((min_plot_time,max_plot_time))
ax0.invert_yaxis()
ax0.set_xlabel('Thickness (m)')
ax0.set_ylabel('Time (s)')
plt.text(2,
        min_plot_time + (lyr_times[0,0] - min_plot_time)/2.,
        'Layer 1',
        fontsize=16)
plt.text(dz_max/dz_step - 2,
        lyr_times[-1,0] + (lyr_times[-1,1] - lyr_times[-1,0])/2.,
        'Layer 2',
        fontsize=16,
        horizontalalignment='right')
plt.text(2,
        lyr_times[0,0] + (max_plot_time - lyr_times[0,0])/2.,
        'Layer 3',
        fontsize=16)
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
ax0.set_xlim((-excursion, ntrc+excursion))

ax1 = fig.add_subplot(gs[1])
plot_vawig(ax1, syn_zo, t, excursion, highlight=tuning_trace)
ax1.plot(lyr_times[:,0], color='blue', lw=1.5)
ax1.plot(lyr_times[:,1], color='red', lw=1.5)
ax1.set_ylim((min_plot_time,max_plot_time))
ax1.invert_yaxis()
ax1.set_xlabel('Thickness (m)')
ax1.set_ylabel('Time (s)')

ax2 = fig.add_subplot(gs[2])
ax2.plot(syn_zo[:,lyr_indx[:,0]], color='blue')
ax2.set_xlim((-excursion, ntrc+excursion))
ax2.axvline(tuning_trace, color='k', lw=2)
ax2.grid()
ax2.set_title('Upper interface amplitude')
ax2.set_xlabel('Thickness (m)')
ax2.set_ylabel('Amplitude')
plt.text(tuning_trace + 2,
        plt.ylim()[0] * 1.1,
        'Tuning thickness = {0} m'.format(str(tuning_thickness)),
        fontsize=16)

st.pyplot(fig)

st.success('Analysis complete!')

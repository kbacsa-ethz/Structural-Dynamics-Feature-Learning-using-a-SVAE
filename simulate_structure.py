import os
import lmdb
import random
import numpy as np
import pyarrow as pa
from scipy.signal import detrend
from scipy.integrate import odeint
from scipy.interpolate import interp1d


# Function to read raw data from a file
def raw_reader(path):
    with open(path, 'rb') as ff:
        bin_data = ff.read()
    return bin_data


# Function to serialize an object using PyArrow
def dumps_pyarrow(obj):
    return pa.serialize(obj).to_buffer()


# Define the Duffing oscillator differential equation
def duffing(w, t, p):
    m, c, k, k3, ext = p

    ndof = m.shape[0]
    A = np.concatenate(
        [
            np.concatenate([np.zeros([ndof, ndof]), np.eye(ndof)], axis=1),  # Link velocities
            np.concatenate([-np.linalg.solve(m, k), -np.linalg.solve(m, c)], axis=1),  # Movement equations
        ], axis=0)

    B = np.concatenate(
        [
            np.concatenate([np.zeros([ndof, ndof]), np.zeros([ndof, ndof])], axis=1),
            np.concatenate([-np.linalg.solve(m, k3), np.zeros([ndof, ndof])], axis=1),  # Duffing term
        ], axis=0)

    nonlinear = np.zeros_like(w)
    nonlinear[:ndof] = w[:ndof] ** 3
    nonlinear = B @ nonlinear
    forcing = np.concatenate([np.zeros(ndof), np.linalg.solve(m, ext(t))])
    func = A @ w + nonlinear + forcing
    return func


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Sampling and simulation parameters
fs = 100
write_frequency = 20
n_samples = 10
seq_len = 300
dt = 1 / fs
excitation_length = int(10 / dt)
band_length = int(2 / dt)

n_iterations = {
    'train': n_samples,
    'val': n_samples // 10,
    'test': n_samples // 10
}

abserr = 1e-8
relerr = 1e-6
signal_snr = 10  # [dB]

# Construct linear system matrices
n_classes = 6
n_dof = 20
m_i = 0.1  # kg
k_l = 1500  # N/m
c_l = 0.1  # kg/s
k_nl = 1.2 * k_l  # N/m^3

# Define masses and spring constants
masses = np.ones(n_dof) * m_i
masses[-1] = 1e1
masses[0] = 1e1
masses[9] = 1e1

spring_linear = np.ones(n_dof + 1) * k_l
spring_linear_diagonal = spring_linear[:-1] + spring_linear[1:]
spring_linear_lu = -spring_linear[1:-1]

spring_nonlinear = np.ones(n_dof + 1) * k_nl
spring_nonlinear_diagonal = spring_nonlinear[:-1] + spring_nonlinear[1:]
spring_nonlinear_lu = -spring_nonlinear[1:-1]

# Create mass and stiffness matrices
M = np.diag(masses)
K = np.diag(spring_linear_diagonal) + np.diag(spring_linear_lu, 1) + np.diag(spring_linear_lu, -1)
K3 = np.diag(spring_nonlinear_diagonal) + np.diag(spring_nonlinear_lu, 1) + np.diag(spring_nonlinear_lu, -1)

# Calculate Rayleigh damping matrix
eigen = np.sqrt(np.abs(np.linalg.eigvals(np.linalg.solve(-M, K))))
sorted(eigen)
damping_ratio = 0.01
zeta = np.ones([len(eigen), 1]) * damping_ratio
damping_matrix = 0.5 * np.stack([1 / eigen, eigen], axis=-1)
ab, res, rank, s = np.linalg.lstsq(damping_matrix, zeta)
alpha = ab[0]
beta = ab[1]
C = alpha * M + beta * K

phases = ['train', 'val', 'test']

# Loop through different phases (train, val, test)
for phase in phases:
    print("Doing phase: {}".format(phase))
    iteration = 0
    count = 0
    keys = []

    # Open LMDB database for the current phase
    db = lmdb.open(os.path.join('.', '{}_dataset'.format(phase)), subdir=True,
                   map_size=1099511627776 * 2,
                   readonly=False, meminit=False, map_async=True)

    txn = db.begin(write=True)

    # Generate labels for the current phase
    phase_labels = np.round(np.linspace(0., n_classes - 1, n_iterations[phase])).astype(int)
    np.random.shuffle(phase_labels)

    # Iterate through the samples for the current phase
    for it in range(n_iterations[phase]):
        print("Simulating system for iteration {}".format(it))

        # Generate white noise input
        min_freq = 1
        max_freq = fs // 2
        fsamples = (max_freq - min_freq) * 10
        freqs = np.abs(np.fft.fftfreq(fsamples, 1 / fs))
        samples = fs * 10
        f = np.zeros(samples)
        idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
        f[idx] = 1
        f = np.array(f, dtype="complex")
        Np = (len(f) - 1) // 2
        angles = np.random.rand(Np) * 2 * np.pi
        angles = np.cos(angles) + 1j * np.sin(angles)
        f[1: Np + 1] *= angles
        f[-1: -1 - Np: -1] = np.conj(f[1: Np + 1])
        noise = np.fft.ifft(f).real
        force_amp = 1e3
        force = noise * force_amp
        n_max = len(force)

        force_input = np.zeros([n_dof, n_max])
        force_input[:, :] = force

        # Make force proportional to mass
        force_input = M @ force_input

        # Simulation parameters
        w0 = np.zeros(2 * n_dof)
        tics = np.linspace(0., n_max * dt, num=n_max, endpoint=False)
        fint = interp1d(tics, force_input, fill_value='extrapolate')

        # Select damage level based on phase_labels
        damage_class = phase_labels[it]

        if damage_class == 0:
            flexibility = 1.0
        elif damage_class == 1:
            flexibility = random.uniform(0.85, 0.95)
        elif damage_class == 2:
            flexibility = random.uniform(0.7, 0.8)
        elif damage_class == 3:
            flexibility = random.uniform(0.55, 0.65)
        elif damage_class == 4:
            flexibility = random.uniform(0.4, 0.5)
        elif damage_class == 5:
            flexibility = random.uniform(0.25, 0.35)
        else:
            raise ValueError

        p = [M, C, K * flexibility, K3 * flexibility, fint]

        # Call the ODE solver (Duffing oscillator)
        wsol = odeint(duffing, w0, tics, args=(p,),
                      atol=abserr, rtol=relerr)

        # Recover acceleration
        wsol_dot = np.zeros_like(wsol)
        for idx, step in enumerate(tics):
            wsol_dot[idx, :] = duffing(wsol[idx, :], step, p)

        # Join states (position and velocity)
        state = np.concatenate([wsol, wsol_dot[:, n_dof:]], axis=1)
        label = damage_class

        # Negative mining: set state to zeros with a low probability
        observations = state.copy()
        if random.random() < 0.05:
            state = np.zeros_like(state)
            observations = np.zeros_like(observations)
        else:
            # add noise
            for i in range(3*n_dof):
                state_signal = detrend(state[:, i])
                signal_power = np.mean(state_signal ** 2)
                signal_power_dB = 10 * np.log10(signal_power)
                noise_dB = signal_power_dB - signal_snr
                noise_watt = 10 ** (noise_dB/10)
                noise = np.random.normal(0, np.sqrt(noise_watt), state.shape[0])
                noise = detrend(noise)  # just in case
                observations[:, i] = state[:, i] + noise

        # Store the data in LMDB
        for i in range(state.shape[0] // seq_len):
            key = (iteration << 16) + i
            keys.append(key)
            txn.put(u'states_{}'.format(key).encode('ascii'),
                    dumps_pyarrow(state[seq_len * i:seq_len * (i + 1), :]))
            txn.put(u'observations_{}'.format(key).encode('ascii'),
                    dumps_pyarrow(observations[seq_len * i:seq_len * (i + 1), :]))
            txn.put(u'force_{}'.format(key).encode('ascii'), dumps_pyarrow(force[seq_len * i: seq_len * (i + 1)]))
            txn.put(u'iteration_{}'.format(key).encode('ascii'), dumps_pyarrow(it))
            txn.put(u'tmin_{}'.format(key).encode('ascii'), dumps_pyarrow(i * seq_len))
            txn.put(u'tmax_{}'.format(key).encode('ascii'), dumps_pyarrow((i + 1) * seq_len))
            txn.put(u'label_{}'.format(key).encode('ascii'), dumps_pyarrow(label))
            txn.put(u'flexibility_{}'.format(key).encode('ascii'), dumps_pyarrow(flexibility))

            if count % write_frequency == 0:
                print("[%d]" % count)
                txn.commit()
                txn = db.begin(write=True)

            count += 1
        iteration += 1

    # Finish iterating through the dataset
    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

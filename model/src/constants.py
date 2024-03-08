# Zenke's paper uses a theta_rest of -50mV
THETA_REST = 0

# Zenke's paper uses a lambda of 1e-4 (fixed in erratum)
LAMBDA_HEBBIAN = 1e-4

# Zenke's paper uses a beta of -1mV
ZENKE_BETA = 1

# Zenke's paper uses a xi of 1e-7 (fixed in erratum)
XI = 1e-7

# Zenke's paper uses a delta of 1e-3 (fixed in erratum)
DELTA = 1e-3

# Zenke's paper uses a kappa of 10 Hz
KAPPA = 10

# Zenke's paper uses a tau_mean of 600s
TAU_MEAN = 600
# Zenke's paper uses a tau_var of 20ms
TAU_VAR = .020
# Zenke's paper uses a tau_stdp of 20ms
TAU_STDP = .020
# Zenke's paper uses a .1ms time step
DT = .1

# Zenke's paper uses tau_rise and tau_fall of these values in units of ms
TAU_RISE_ALPHA = .002
TAU_FALL_ALPHA = .010
TAU_RISE_EPSILON = .005
TAU_FALL_EPSILON = .020

MAX_RETAINED_SPIKES = int(20 / DT)

DATA_MEM_ASSUMPTION = 0.5

DECAY_BETA = 0.85

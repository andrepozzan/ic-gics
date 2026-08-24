import numpy as np

def qam_mod(bits, modulation_order):
    
    k = int(np.log2(modulation_order)) 

    k_half = k // 2
    n_levels = 2**k_half
    
    # Adjust the bit string to be a multiple of k (bits per symbol)
    bits = list(bits)
    if len(bits) % k != 0:
        bits += ['0'] * (k - (len(bits) % k))

    def binary_to_gray(n):
        return n ^ (n >> 1)

    levels = np.arange(n_levels)
    gray_levels = binary_to_gray(levels)
    
    
    qam_levels = 2 * np.arange(n_levels) - (n_levels - 1)
    
    mapping_i = np.zeros(n_levels)
    mapping_q = np.zeros(n_levels)
    
    for i, g in enumerate(gray_levels):
        mapping_i[g] = qam_levels[i]
        mapping_q[g] = qam_levels[n_levels - 1 - i]

    # 5. Bit processing
    symbols = []
    for i in range(0, len(bits), k):
        bit_group = bits[i:i+k]
        
        # Split the group: half for I, half for Q
        bits_i = "".join(bit_group[:k_half])
        bits_q = "".join(bit_group[k_half:])
        
        # Convert to decimal
        idx_i = int(bits_i, 2)
        idx_q = int(bits_q, 2)
        
        # Map to the constellation using Gray mapping
        comp_i = mapping_i[idx_i]
        comp_q = mapping_q[idx_q]
        
        symbols.append(complex(comp_i, comp_q))

    return np.array(symbols)

def qam_demod(symbols, modulation_order):
    
    print("\n \n \n \n")
    
    k = int(np.log2(modulation_order))
    k_half = k // 2
    n_levels = 2 ** k_half

    qam_levels = 2 * np.arange(n_levels) - (n_levels - 1)
    print('qam_levels: ', qam_levels)

    # Inverse maps from detected PAM value to Gray index for each axis.
    inv_map_i = {}
    inv_map_q = {}
    
    for level in range(n_levels):
        gray = level ^ (level >> 1)
        
        inv_map_i[qam_levels[level]] = gray
        inv_map_q[qam_levels[n_levels - 1 - level]] = gray

    # Simple AGC: match received average symbol power to the ideal M-QAM power.
    ideal_symbol_power = (2.0 / 3.0) * (n_levels**2 - 1)
    
    measured_power = np.mean(np.abs(symbols) ** 2)
    if measured_power > 0:
        gain = np.sqrt(ideal_symbol_power / measured_power)
        print('gain: ', gain)
        symbols = symbols * gain
        
        

    I = np.real(symbols)
    Q = np.imag(symbols)

    I_column = I[:, None]
    Q_column = Q[:, None]
    qam_levels_row = qam_levels[None, :]

    #Get indices of the closest constellation points for I and Q components
    minor_distance_i = np.argmin(np.abs(I_column - qam_levels_row), axis=1)
    minor_distance_q = np.argmin(np.abs(Q_column - qam_levels_row), axis=1)

    i_levels = qam_levels[minor_distance_i]
    print('i_levels: ', i_levels)
    q_levels = qam_levels[minor_distance_q]
    print('q_levels: ', q_levels)

    bits = []
    for i_level, q_level in zip(i_levels, q_levels):
        gray_i = int(inv_map_i[i_level])
        gray_q = int(inv_map_q[q_level])
        bits.append(format(gray_i, f'0{k_half}b') + format(gray_q, f'0{k_half}b'))

    return ''.join(bits)



# Modulation QAM: s(t) = I(t)·cos(2πfc·t) − Q(t)·sin(2πfc·t)
def qam_modulate_passband(baseband_signal, fc, fs):
    N = len(baseband_signal)
    t = np.arange(N) / fs  # time vector
    
    # Extract I and Q components
    I = np.real(baseband_signal)
    Q = np.imag(baseband_signal)
    
    carrier_cos = np.cos(2 * np.pi * fc * t)
    carrier_sin = np.sin(2 * np.pi * fc * t)
    
    s_t = I * carrier_cos - Q * carrier_sin
    
    return s_t, t

def qam_demodulate_passband(s_t, fc, fs):
    N = len(s_t)
    t = np.arange(N) / fs
    I_rec = s_t * np.cos(2 * np.pi * fc * t) * 2
    Q_rec = -s_t * np.sin(2 * np.pi * fc * t) * 2
    
    return I_rec + 1j * Q_rec

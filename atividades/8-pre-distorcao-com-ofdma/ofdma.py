import numpy as np

from qam import qam_mod, qam_demod

def generate_OFDMA_signal(Nfft, number_of_users, subcarriers_per_user, modulation_order):
    # Complex subcarrier grid (size Nfft).
    
    complete_band = np.zeros(Nfft, dtype=complex)

    # Store original user bit streams for optional BER checks.
    all_original_bits = []

    for user in range(number_of_users):
        start = user * subcarriers_per_user
        end = start + subcarriers_per_user

        # Generate random user bits and map them to QAM symbols.
        bits = ''.join(np.random.choice(['0','1'], int(np.log2(modulation_order))*subcarriers_per_user))
        # bits = "0000 0001 0010 0011 0100";
        bits = bits.replace(" ", "")
        
        symbols = qam_mod(bits, modulation_order)
        print("USER", user, "BITS:", bits)

        # Assign user symbols to contiguous subcarriers.
        complete_band[start:end] = symbols
        
        # Keep the original bit stream.
        all_original_bits.append(bits)

    # Convert frequency-domain grid to time-domain signal.
    band_ifft = np.fft.ifft(complete_band) * np.sqrt(Nfft)    

    return band_ifft, all_original_bits

def extract_bits_from_ofdma(received_signal, Nfft, number_of_users, subcarriers_per_user, modulation_order):
    recovered_band = np.fft.fft(received_signal) / np.sqrt(Nfft) # Common OFDM scaling adjustment
    
    
    all_recovered_bits = []
    for user in range(number_of_users):
        start = user * subcarriers_per_user
        end = start + subcarriers_per_user
        user_symbols = recovered_band[start:end]
        
        # 3. Demodulate
        user_bits = qam_demod(user_symbols, modulation_order)
        all_recovered_bits.append(user_bits)
        
    return all_recovered_bits 
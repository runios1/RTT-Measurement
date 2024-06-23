import numpy as np

def generate_message(d, method):
    """ Generate a message with data bits and parity bits """
    data = np.random.randint(0, 2, d)
    if method == 'single_parity':
        parity = np.sum(data) % 2
        return np.append(data, parity)
    elif method == 'parity_matrix':
        sqrt_d = int(np.sqrt(d))
        data_matrix = data.reshape((sqrt_d, sqrt_d))
        row_parity = np.sum(data_matrix, axis=1) % 2
        col_parity = np.sum(data_matrix, axis=0) % 2
        parity_bits = np.append(row_parity, col_parity)
        return np.append(data, parity_bits)

def flip_bits(message, p):
    """ Simulate the noisy channel by flipping bits with probability p """
    noise = np.random.rand(len(message)) < p
    return np.bitwise_xor(message, noise.astype(int))

def detect_and_correct(message, method):
    """ Detect and correct errors using parity bits """
    if method == 'single_parity':
        data = message[:-1]
        received_parity = message[-1]
        calculated_parity = np.sum(data) % 2
        if received_parity == calculated_parity:
            return data, 'success'
        else:
            return None, 'retransmit'
    elif method == 'parity_matrix':
        d = int(len(message) - 2 * int(np.sqrt(len(message) - int(np.sqrt(len(message))))))
        data = message[:d]
        sqrt_d = int(np.sqrt(d))
        data_matrix = data.reshape((sqrt_d, sqrt_d))
        received_row_parity = message[d:d + sqrt_d]
        received_col_parity = message[d + sqrt_d:]
        calculated_row_parity = np.sum(data_matrix, axis=1) % 2
        calculated_col_parity = np.sum(data_matrix, axis=0) % 2
        if np.array_equal(received_row_parity, calculated_row_parity) and np.array_equal(received_col_parity, calculated_col_parity):
            return data, 'success'
        else:
            return None, 'retransmit'

def simulate(d, p, method, trials=1000):
    success_count = 0
    transmitted_bits = 0
    for _ in range(trials):
        message = generate_message(d, method)
        transmitted_bits += len(message)
        noisy_message = flip_bits(message, p)
        data, status = detect_and_correct(noisy_message, method)
        if status == 'success':
            success_count += d
    efficiency_factor = success_count / transmitted_bits
    return efficiency_factor

def main():
    d_values = [484, 576]  # Use perfect squares
    p_values = [0.0001, 0.001, 0.01, 0.05]
    methods = ['single_parity', 'parity_matrix']
    for d in d_values:
        for p in p_values:
            for method in methods:
                efficiency = simulate(d, p, method)
                print(f"d = {d}, p = {p}, method = {method}, efficiency = {efficiency:.4f}")

if __name__ == "__main__":
    main()

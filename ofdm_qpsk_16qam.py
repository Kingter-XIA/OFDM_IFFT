import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class QPSK_OFDM_Modulator:
    def __init__(self,
                 modulation="QPSK",          
                 symbol_duration=1.0,
                 sampling_rate=1024,
                 n_ifft=64):

        self.modulation = modulation.upper()

        # -------------------------------
        # QPSK Gray 映射
        s = 1 / np.sqrt(2)
        self.qpsk_map = {
            (0, 0):  s + 1j * s,
            (0, 1): -s + 1j * s,
            (1, 1): -s - 1j * s,
            (1, 0):  s - 1j * s
        }

        # -------------------------------
        # 16QAM Gray 映射（归一化）
        a = 1 / np.sqrt(10)
        self.qam16_map = {
            (0,0,0,0): (-3 + 3j)*a,
            (0,0,0,1): (-3 + 1j)*a,
            (0,0,1,1): (-3 - 1j)*a,
            (0,0,1,0): (-3 - 3j)*a,

            (0,1,0,0): (-1 + 3j)*a,
            (0,1,0,1): (-1 + 1j)*a,
            (0,1,1,1): (-1 - 1j)*a,
            (0,1,1,0): (-1 - 3j)*a,

            (1,1,0,0): ( 1 + 3j)*a,
            (1,1,0,1): ( 1 + 1j)*a,
            (1,1,1,1): ( 1 - 1j)*a,
            (1,1,1,0): ( 1 - 3j)*a,

            (1,0,0,0): ( 3 + 3j)*a,
            (1,0,0,1): ( 3 + 1j)*a,
            (1,0,1,1): ( 3 - 1j)*a,
            (1,0,1,0): ( 3 - 3j)*a,
        }

        if self.modulation == "QPSK":
            self.bits_per_symbol = 2
            self.map = self.qpsk_map
        elif self.modulation == "16QAM":
            self.bits_per_symbol = 4
            self.map = self.qam16_map
        else:
            raise ValueError("modulation must be 'QPSK' or '16QAM'")

        # -------------------------------
        self.carrier_freqs = np.arange(1, 9)
        self.symbol_duration = symbol_duration
        self.sampling_rate = sampling_rate

        self.t_symbol = np.linspace(
            0, symbol_duration,
            int(sampling_rate * symbol_duration),
            endpoint=False
        )

        self.N_ifft = n_ifft
        self.carrier_indices = np.arange(1, 9)

        self.carrier_bits = {}
        self.carrier_symbols = {}
        self.carrier_signals = {}
        self.combined = None
        self.plot_symbols = 4 

    # --------------------------------------------------
    def bytes_to_bits(self, data_bytes):
        bits = []
        for b in data_bytes:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        return np.array(bits, dtype=int)

    def map_bits_to_symbol(self, bits):
        return self.map[tuple(bits)]

    def generate_symbol_wave(self, freq, symbol):
        t = self.t_symbol
        I = np.real(symbol)
        Q = np.imag(symbol)
        return I * np.cos(2*np.pi*freq*t) - Q * np.sin(2*np.pi*freq*t)

    # --------------------------------------------------
    def modulate(self, data_bytes):
        bits = self.bytes_to_bits(data_bytes)

        carrier_bits, carrier_symbols, carrier_signals = {}, {}, {}

        symbols_per_carrier = 8 // self.bits_per_symbol

        for i, freq in enumerate(self.carrier_freqs):
            byte_bits = bits[i*8:i*8+8]
            carrier_bits[freq] = byte_bits

            symbols = []
            for k in range(0, 8, self.bits_per_symbol):
                symbols.append(
                    self.map_bits_to_symbol(byte_bits[k:k+self.bits_per_symbol])
                )

            # 若符号数不足 4（16QAM 情况），循环补齐
            while len(symbols) < self.plot_symbols:
                symbols.append(symbols[len(symbols) % len(symbols)])

            carrier_symbols[freq] = symbols

            print(f"[{self.modulation}] {freq}Hz bits={byte_bits} "
                  f"symbols={[f'{s.real:.2f}{s.imag:+.2f}j' for s in symbols]}")

            sig = np.concatenate([
                self.generate_symbol_wave(freq, s) for s in symbols
            ])
            carrier_signals[freq] = sig

        samples_per_symbol = len(self.t_symbol)
        total_samples = samples_per_symbol * self.plot_symbols
        t_total = np.linspace(0,
                              self.plot_symbols*self.symbol_duration,
                              total_samples,
                              endpoint=False)

        combined = np.zeros(total_samples)
        for s in carrier_signals.values():
            combined += s

        self.carrier_bits = carrier_bits
        self.carrier_symbols = carrier_symbols
        self.carrier_signals = carrier_signals
        self.combined = combined

        return carrier_bits, carrier_symbols, carrier_signals, combined, t_total

    # --------------------------------------------------
    def ifft(self, carrier_symbols):
        symbols_per_carrier = len(next(iter(carrier_symbols.values())))
        ifft_inputs, ifft_outputs = [], []

        for sym_idx in range(self.plot_symbols):
            freq_domain = np.zeros(self.N_ifft, dtype=complex)

            for i, freq in enumerate(self.carrier_freqs):
                idx = self.carrier_indices[i]
                freq_domain[idx] = carrier_symbols[freq][sym_idx]

            for i in range(1, 9):
                freq_domain[self.N_ifft - i] = np.conj(freq_domain[i])

            ifft_inputs.append(freq_domain)
            ifft_outputs.append(np.fft.ifft(freq_domain))

        return ifft_inputs, ifft_outputs

    # --------------------------------------------------
    def plot_all(self, carrier_bits, carrier_symbols, carrier_signals,
                 combined_signal, t_total, ifft_inputs, ifft_outputs):
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(4, 1, figure=fig)
        gs = GridSpec(4, 1, figure=fig, height_ratios=[2, 1, 1, 1])                     # ax1 高度是其他的2倍
    
        # 各载波波形
        ax1 = fig.add_subplot(gs[0, :])
        offset = 2.5
        for i, f in enumerate(self.carrier_freqs):
            ax1.plot(t_total, carrier_signals[f] + i * offset)                          #绘曲线
            ax1.text(t_total[-1] + 0.05, i * offset,                                    #右侧显示对应的比特
                     ''.join(str(b) for b in carrier_bits[f])+'b',ha='left', va='top',
                     fontsize=12)
            ax1.text(-0, i * offset, f"{f}Hz", ha='right', va='top', fontsize=12)       #左侧显示对应的频率

        ax1.set_title(f"各载波{self.modulation}调制信号（4符号周期）")
        ax1.set_xlabel("时间 (s)")
        ax1.set_ylabel("幅度(偏移显示)")
        ax1.grid(True)

        # 合成信号
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(t_total, combined_signal)
        ax2.set_title("8个载波叠加信号")
        ax2.set_xlabel("时间 (s)")
        ax2.grid(True)

        # IFFT输出: 我们只显示实部（振幅）
        ax4 = fig.add_subplot(gs[2,:])
        out_concat = np.concatenate(ifft_outputs)
        ax4.scatter(np.arange(len(out_concat)), np.real(out_concat),
           color='blue', s=10, marker='o', linewidth=0.5,
           label=f"实部")

        ax4.set_title(f"IFFT输出（时域采样点，{self.N_ifft}点/符号）")
        ax4.legend()
        ax4.grid(True)

        #叠加信号
        ax5 = fig.add_subplot(gs[3, :])
        combined_signal_norm = combined_signal / np.max(np.abs(combined_signal))
        out_concat_real = np.real(out_concat)
        out_concat_real_norm = out_concat_real / np.max(np.abs(out_concat_real))
        M = len(combined_signal_norm)
        N = len(out_concat_real_norm)
        t_high_res = np.linspace(0, 4 * self.symbol_duration, M, endpoint=False)
        t_low_res = np.linspace(0, 4 * self.symbol_duration, N, endpoint=False)

        ax5.plot(t_high_res, combined_signal_norm, 
                 color='red', linestyle='-', alpha=0.7, linewidth=2, 
                 label="8载波合成信号")
        ax5.scatter(t_low_res, out_concat_real_norm, 
                    color='blue', s=10, marker='o', linewidth=0.5,
                    label=f"IFFT输出点 ({N}点)")
        ax5.set_title("IFFT输出与多载波合成信号对比")
        ax5.set_xlabel("时间 (s)")
        ax5.set_ylabel("归一化幅度")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# --------------------------------------------------
def main():
    s = "元旦快乐"
    data_bytes = list(s.encode('gbk'))

    print("QPSK 演示")
    mod = QPSK_OFDM_Modulator(modulation="QPSK")
    cbits, csym, csig, combined, t = mod.modulate(data_bytes)
    ifft_in, ifft_out = mod.ifft(csym)
    mod.plot_all(cbits, csym, csig, combined, t, ifft_in, ifft_out)

    print("\n16QAM 演示")
    mod = QPSK_OFDM_Modulator(modulation="16QAM")
    cbits, csym, csig, combined, t = mod.modulate(data_bytes)
    ifft_in, ifft_out = mod.ifft(csym)
    mod.plot_all(cbits, csym, csig, combined, t, ifft_in, ifft_out)

if __name__ == "__main__":
    main()

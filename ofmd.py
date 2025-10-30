import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class QPSK_OFDM_Modulator:
    def __init__(self, symbol_duration=1.0, sampling_rate=1024, n_ifft=64):
        # 定义Gray编码的QPSK星座点(I + jQ)
        s = 1 / np.sqrt(2)
        self.qpsk_map = {
            (0, 0): s + 1j * s,
            (0, 1): -s + 1j * s,
            (1, 1): -s - 1j * s,
            (1, 0): s - 1j * s
        }

        self.carrier_freqs = np.arange(1, 9)            # 1~8Hz载频
        self.symbol_duration = symbol_duration          # 一个符号周期是1s对应1Hz载频的1个周期，8Hz载频的8个周期
        self.sampling_rate = sampling_rate              # 时域信号采用速率，用于绘图显示
        self.t_symbol = np.linspace(0,                  # 每个采样时间点
                                    symbol_duration,
                                    int(sampling_rate * symbol_duration),
                                    endpoint=False)
        
        # n_ifft 需要大于8。 我们用1Hz~8Hz作为载波，如果采用8点IFFT,其默认是0-7Hz，而我们为了展示目的，不想用DC。
        # 使用16点及以上的IFFT，我们只需要取输出中对应的1-8Hz的索引对应的值用于展示和对比，就可以。
        self.N_ifft = n_ifft
        
        self.carrier_indices = np.arange(1, 9)      # 在IFFT中，索引1~8对应1~8Hz
                
        self.carrier_bits = {}                      #每个载频的比特流例如[0,1,0,1,0,1,0]
        self.carrier_symbols= {}                    #每个载频的调制符号 2bit -> 1 符号，共4个
        self.carrier_signals = {}                   #每个载频的时域波形
        self.combined =None                         #合成的时域波形

    def bytes_to_bits(self, data_bytes):
        #将8个字节转为64个bit流，按MSB->LSB顺序处理
        bits = []
        for b in data_bytes:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        return np.array(bits, dtype=int)

    def qpsk_pair(self, pair):
        #每2个bit生成一个QPSK调制幅度和相位
        return self.qpsk_map[(pair[0], pair[1])]

    def generate_symbol_wave(self, freq, symbol):
        #生成一个符号周期的波形 s(t)=I*cos- Q*sin
        t = self.t_symbol
        I = np.real(symbol) 
        Q = np.imag(symbol)
        return I * np.cos(2 * np.pi * freq * t) - Q * np.sin(2 * np.pi * freq * t)

    def modulate(self, data_bytes):
        #输入是8字节，即64比特流
        bits = self.bytes_to_bits(data_bytes)

        #每个载波使用 1 字节（8 bit=4 符号）如果不足，则padding
        if len(data_bytes) < 8:
            pad = [0] * (8 - len(data_bytes))
            data_bytes += pad
            
        carrier_bits, carrier_symbols, carrier_signals = {}, {}, {}

        for i, freq in enumerate(self.carrier_freqs):            
            byte_bits = bits[i*8:i*8+8]             

            #每个载频的比特流例如[0,1,0,1,0,1,0]
            carrier_bits[freq] = np.array(byte_bits, dtype=int)
            
            #每个载频的调制符号 2bit -> 1 符号，共4个
            symbols = [self.qpsk_pair(byte_bits[k:k + 2]) for k in range(0, 8, 2)]
            carrier_symbols[freq] = symbols
            print(f'byte_bits={byte_bits} symbols={[f"{x.real:.3f}{x.imag:+.3f}j" for x in symbols]}')
            # 生成和拼接4个符号周期的时域波形
            sig = np.concatenate([self.generate_symbol_wave(freq, s) for s in symbols])
            carrier_signals[freq] = sig

        samples_per_symbol = len(self.t_symbol)
        total_samples = samples_per_symbol * 4
        t_total = np.linspace(0, 4 * self.symbol_duration, total_samples, endpoint=False)
        
        # 叠加所有载波
        combined = np.zeros(total_samples)
        for s in carrier_signals.values():
            combined += s

        self.carrier_bits = carrier_bits
        self.carrier_symbols = carrier_symbols
        self.carrier_signals = carrier_signals
        self.combined = combined
        return carrier_bits, carrier_symbols, carrier_signals, combined, t_total#, ifft_inputs, ifft_outputs
        
    def ifft(self, carrier_symbols):
        # 在每个符号周期内，按符号周期进行IFFT
        # 16点IFFT输入：每个符号周期对应16点频域向量
        ifft_inputs, ifft_outputs = [], []
        
        for sym_idx in range(4):
            # 创建16点频域向量
            freq_domain = np.zeros(self.N_ifft, dtype=complex)
            
            # 将1~8Hz的符号放到对应的索引位置 (索引1~8) idx:1~8, i:0~7, freq:1~8
            for i, freq in enumerate(self.carrier_freqs):
                idx = self.carrier_indices[i]
                freq_domain[idx] = carrier_symbols[freq][sym_idx]
            
            # 设置Hermitian对称以获得实信号
            # 对于实信号，频域必须满足共轭对称性
            for i in range(1, 9):
                freq_domain[self.N_ifft - i] = np.conj(freq_domain[i])
            
            ifft_inputs.append(freq_domain)
            ifft_output = np.fft.ifft(freq_domain)
            ifft_outputs.append(ifft_output)

        return ifft_inputs, ifft_outputs

    # --------------------------------------------------
    def plot_all(self, carrier_bits, carrier_symbols, carrier_signals,
                 combined_signal, t_total, ifft_inputs, ifft_outputs):
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(4, 1, figure=fig)
        gs = GridSpec(4, 1, figure=fig, height_ratios=[2, 1, 1, 1])  # ax1 高度是其他的2倍
    
        # 各载波波形
        ax1 = fig.add_subplot(gs[0, :])
        offset = 2.5
        for i, f in enumerate(self.carrier_freqs):
            ax1.plot(t_total, carrier_signals[f] + i * offset)                          #绘曲线
            ax1.text(t_total[-1] + 0.05, i * offset,                                    #右侧显示对应的比特
                     ''.join(str(b) for b in carrier_bits[f])+'b',ha='left', va='top',
                     fontsize=12)
            ax1.text(-0, i * offset, f"{f}Hz", ha='right', va='top', fontsize=12)       #左侧显示对应的频率

        ax1.set_title("各载波QPSK调制信号（4符号周期）")
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
    print(f'data_bytes={[f"0x{byte:02X}" for byte in data_bytes]}')
    mod = QPSK_OFDM_Modulator()
    cbits, csym, csig, combined, t= mod.modulate(data_bytes)
    ifft_in, ifft_out = mod.ifft(csym)
    mod.plot_all(cbits, csym, csig, combined, t, ifft_in, ifft_out)

if __name__ == "__main__":
    main()
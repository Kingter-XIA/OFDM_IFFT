import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class QPSK_1Hz_Visualizer:
    def __init__(self, symbol_duration=1.0, sampling_rate=100):
        # QPSK星座点映射 (Gray编码)
        s = 1 / np.sqrt(2)
        self.qpsk_map = {
            (0, 0): (s, s),      # 00 -> 45° (I=+, Q=+)
            (0, 1): (-s, s),     # 01 -> 135° (I=-, Q=+)
            (1, 1): (-s, -s),    # 11 -> 225° (I=-, Q=-)
            (1, 0): (s, -s)      # 10 -> 315° (I=+, Q=-)
        }
        
        self.symbol_duration = symbol_duration
        self.sampling_rate = sampling_rate
        self.carrier_freq = 1.0  # 1Hz载波频率
        
        # 单个符号的时间轴
        self.t_symbol = np.linspace(0, symbol_duration, int(sampling_rate * symbol_duration), endpoint=False)
        
        # 4个符号的连续时间轴
        self.t_total = np.linspace(0, 4 * symbol_duration, 4 * int(sampling_rate * symbol_duration), endpoint=False)
    
    def generate_qpsk_symbol(self, bit_pair):
        I, Q = self.qpsk_map[bit_pair]
        
        # 生成载波
        carrier_I = I * np.cos(2 * np.pi * self.carrier_freq * self.t_symbol)
        carrier_Q = Q * np.sin(2 * np.pi * self.carrier_freq * self.t_symbol)
        
        # QPSK调制信号
        modulated_signal = carrier_I - carrier_Q
        
        return modulated_signal, carrier_I, carrier_Q, I, Q
    
    def generate_continuous_qpsk(self, symbol_sequence):
        #生成连续的QPSK信号
        continuous_signal = np.array([])
        I_components = np.array([])
        Q_components = np.array([])
        
        for bit_pair in symbol_sequence:
            modulated, carrier_I, carrier_Q, I, Q = self.generate_qpsk_symbol(bit_pair)
            continuous_signal = np.concatenate([continuous_signal, modulated])
            I_components = np.concatenate([I_components, carrier_I])
            Q_components = np.concatenate([Q_components, carrier_Q])
        
        return continuous_signal, I_components, Q_components
    
    def plot_continuous_qpsk(self):
        #绘制连续的4个QPSK符号波形
        # 定义4个符号的序列
        symbol_sequence = [(0, 0), (0, 1), (1, 1), (1, 0)]
        symbol_labels = ["00", "01", "11", "10"]
        
        # 生成连续信号
        continuous_signal, I_comp, Q_comp = self.generate_continuous_qpsk(symbol_sequence)
        
        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1])
        
        # 主图：连续的QPSK调制信号
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.t_total, continuous_signal, 'b-', linewidth=2, label='QPSK调制信号')
        
        # 标记符号边界和符号类型
        for i in range(4):
            # 符号边界
            ax1.axvline(x=i * self.symbol_duration, color='red', linestyle='--', alpha=0.7)
            
            # 符号标签
            symbol_time = (i + 0.5) * self.symbol_duration
            ax1.text(symbol_time, ax1.get_ylim()[1] * 0.9, symbol_labels[i],
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
            
            # 相位信息
            I, Q = self.qpsk_map[symbol_sequence[i]]
            phase = np.arctan2(Q, I) * 180 / np.pi
            if phase < 0:
                phase += 360
            ax1.text(symbol_time, ax1.get_ylim()[0] * 0.9, f'{phase:.0f}°',
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
        
        ax1.set_title('QPSK调制: 00011011b的4个符号连续波形', fontsize=16, fontweight='bold')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('幅度')
        ax1.grid(False, alpha=0.3)
        ax1.legend()
        
        # 子图1：I路分量
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(self.t_total, I_comp, 'g-', linewidth=1.5, label='I路分量')
        
        # 标记符号边界
        for i in range(4):
            ax2.axvline(x=i * self.symbol_duration, color='red', linestyle='--', alpha=0.7)
            ax2.text((i + 0.5) * self.symbol_duration, ax2.get_ylim()[1] * 0.8, symbol_labels[i],
                    ha='center', va='center', fontsize=10)
        
        ax2.set_title('同相分量 (I)')
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('幅度')
        ax2.grid(False, alpha=0.3)
        ax2.legend()
        
        # 子图2：Q路分量
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(self.t_total, Q_comp, 'r-', linewidth=1.5, label='Q路分量')
        
        # 标记符号边界
        for i in range(4):
            ax3.axvline(x=i * self.symbol_duration, color='red', linestyle='--', alpha=0.7)
            ax3.text((i + 0.5) * self.symbol_duration, ax3.get_ylim()[1] * 0.8, symbol_labels[i],
                    ha='center', va='center', fontsize=10)
        
        ax3.set_title('正交分量 (Q)')
        ax3.set_xlabel('时间 (秒)')
        ax3.set_ylabel('幅度')
        ax3.grid(False, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
    
    
    def plot_constellation(self):
        #绘制QPSK星座图
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制星座点
        for bit_pair, (I, Q) in self.qpsk_map.items():
            label = f"{bit_pair[0]}{bit_pair[1]}b"
            phase = np.arctan2(Q, I) * 180 / np.pi
            if phase < 0:
                phase += 360
            
            ax.scatter(I, Q, s=200, zorder=5, label=f'{label} ({phase:.0f}°)')
            ax.text(I, Q + 0.05, label, ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 绘制单位圆
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta)/np.sqrt(2), np.sin(theta)/np.sqrt(2), 'k--', alpha=0.5)
        
        # 绘制坐标轴
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        
        # 设置图形属性
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('同相分量 (I)')
        ax.set_ylabel('正交分量 (Q)')
        ax.set_aspect('equal')
        #ax.grid(False, alpha=0.3)
        ax.set_title('QPSK星座图', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    visualizer = QPSK_1Hz_Visualizer(symbol_duration=1.0, sampling_rate=1000)
    
    print("QPSK调制波形可视化 (1Hz载波)")
    print("=" * 50)
    print("每个符号持续1秒，使用1Hz载波，每个符号包含一个完整的正弦周期")
    print("符号序列: 00 → 01 → 11 → 10")
    print("对应相位: 45° → 135° → 225° → 315°")
    print()
    
    # 显示星座图
    print("显示QPSK星座图...")
    visualizer.plot_constellation()
    
    # 显示连续波形
    print("显示4个符号的连续波形...")
    visualizer.plot_continuous_qpsk()
    
    
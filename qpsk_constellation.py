import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class QPSK_Waveform_Visualizer:
    def __init__(self, symbol_duration=1.0, sampling_rate=1000, carrier_freq=5):
        # QPSK星座点映射
        s = 1 / np.sqrt(2)
        self.qpsk_map = {
            (0, 0): (s, s),      # 00 -> 45° (I=+, Q=+)
            (0, 1): (-s, s),     # 01 -> 135° (I=-, Q=+)
            (1, 1): (-s, -s),    # 11 -> 225° (I=-, Q=-)
            (1, 0): (s, -s)      # 10 -> 315° (I=+, Q=-)
        }
        
        self.symbol_duration = symbol_duration
        self.sampling_rate = sampling_rate
        self.carrier_freq = carrier_freq
        
        # 时间轴（一个符号周期）
        self.t = np.linspace(0, symbol_duration, int(sampling_rate * symbol_duration), endpoint=False)
    
    def generate_qpsk_waveform(self, bit_pair):
        """生成QPSK调制波形"""
        I, Q = self.qpsk_map[bit_pair]
        
        # 生成载波
        carrier_I = I * np.cos(2 * np.pi * self.carrier_freq * self.t)
        carrier_Q = Q * np.sin(2 * np.pi * self.carrier_freq * self.t)
        
        # QPSK调制信号
        modulated_signal = carrier_I - carrier_Q
        
        return modulated_signal, carrier_I, carrier_Q, I, Q
    
    def plot_qpsk_symbols(self):
        # 4个QPSK符号的比特对
        symbols = [
            ((0, 0), "00b"),
            ((0, 1), "01b"), 
            ((1, 1), "11b"),
            ((1, 0), "10b")
        ]
        
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1], width_ratios=[1, 3])
        
        # 颜色方案
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, ((bit_pair, label), color) in enumerate(zip(symbols, colors)):
            # 生成波形
            modulated, carrier_I, carrier_Q, I, Q = self.generate_qpsk_waveform(bit_pair)
            
            # 主波形图（左列）
            ax_wave = fig.add_subplot(gs[i, 1])
            
            # 绘制调制后的信号
            ax_wave.plot(self.t, modulated, color=color, linewidth=2)#, label=f'调制信号')
            
            # 绘制I和Q分量（半透明）
            #ax_wave.plot(self.t, carrier_I, color='blue', alpha=0.5, linestyle='--', linewidth=1, label='I分量')
            #ax_wave.plot(self.t, carrier_Q, color='red', alpha=0.5, linestyle='--', linewidth=1, label='Q分量')
            
            # 设置标题和标签
            #ax_wave.set_title(f'QPSK符号: {label}', fontsize=12, fontweight='bold')
            #ax_wave.set_xlabel('时间 (s)')
            #ax_wave.set_ylabel('幅度')
            ax_wave.grid(True, alpha=0.3)
            #ax_wave.legend(loc='upper right')
            
            # 添加相位信息
            phase = np.arctan2(Q, I) * 180 / np.pi
            if phase < 0:
                phase += 360
            ax_wave.text(0.02, 0.95, f'符号:{label}   相位: {phase:.0f}° ', 
                        transform=ax_wave.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
            print(f'label={label}')
            # 星座图（右列）
            ax_const = fig.add_subplot(gs[i, 0])
            
            # 绘制星座点
            ax_const.scatter(I, Q, color=color, s=200, zorder=5, edgecolor='black', linewidth=2)
            
            # 绘制参考圆
            theta = np.linspace(0, 2*np.pi, 100)
            ax_const.plot(np.cos(theta)/np.sqrt(2), np.sin(theta)/np.sqrt(2), 
                         'k--', alpha=0.3, linewidth=1)
            
            # 绘制坐标轴
            ax_const.axhline(0, color='black', linewidth=0.5)
            ax_const.axvline(0, color='black', linewidth=0.5)
            
            # 设置星座图范围和标签
            ax_const.set_xlim(-1.2, 1.2)
            ax_const.set_ylim(-1.2, 1.2)
            ax_const.set_xlabel('同相分量 (I)')
            ax_const.set_ylabel('正交分量 (Q)')
            ax_const.set_aspect('equal')
            ax_const.grid(True, alpha=0.3)
            ax_const.set_title('星座点')
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self):
        """在单个图中比较所有4个符号的波形"""
        # 4个QPSK符号
        symbols = [(0, 0), (0, 1), (1, 1), (1, 0)]
        labels = ["00 (45°)", "01 (135°)", "11 (225°)", "10 (315°)"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        for i, (bit_pair, label, color) in enumerate(zip(symbols, labels, colors)):
            # 生成波形
            modulated, _, _, I, Q = self.generate_qpsk_waveform(bit_pair)
            
            # 绘制波形
            axes[i].plot(self.t, modulated, color=color, linewidth=2)
            
            # 计算并显示相位
            phase = np.arctan2(Q, I) * 180 / np.pi
            if phase < 0:
                phase += 360
            
            # 设置标题和标签
            axes[i].set_title(f'QPSK符号 {label} - 相位: {phase:.0f}°', fontweight='bold')
            axes[i].set_ylabel('幅度')
            axes[i].grid(True, alpha=0.3)
            
            # 在最后一个子图添加x轴标签
            if i == 3:
                axes[i].set_xlabel('时间 (s)')
        
        plt.tight_layout()
        plt.show()

    def plot_with_phase_annotation(self):
        """绘制带有相位标注的波形"""
        # 4个QPSK符号
        symbols = [(0, 0), (0, 1), (1, 1), (1, 0)]
        labels = ["00", "01", "11", "10"]
        phases = [45, 135, 225, 315]  # 对应的相位（度）
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        for i, (bit_pair, label, phase, color) in enumerate(zip(symbols, labels, phases, colors)):
            # 生成波形
            modulated, _, _, _, _ = self.generate_qpsk_waveform(bit_pair)
            
            # 绘制波形
            axes[i].plot(self.t, modulated, color=color, linewidth=2.5)
            
            # 添加相位标记线
            # 找到波形的峰值点
            peak_idx = np.argmax(modulated)
            peak_time = self.t[peak_idx]
            peak_value = modulated[peak_idx]
            
            # 绘制峰值标记
            axes[i].axvline(peak_time, color=color, linestyle=':', alpha=0.7)
            axes[i].plot(peak_time, peak_value, 'o', color=color, markersize=8)
            
            # 添加相位文本
            axes[i].text(peak_time + 0.02, peak_value, f'{phase}°', 
                        fontsize=12, fontweight='bold', color=color,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            # 设置标题和标签
            axes[i].set_title(f'QPSK符号: {label}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('幅度', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # 在最后一个子图添加x轴标签
            if i == 3:
                axes[i].set_xlabel('时间 (s)', fontsize=12)
        
        plt.tight_layout()
        plt.suptitle('QPSK调制: 4个符号的波形与相位', fontsize=16, fontweight='bold', y=0.98)
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = QPSK_Waveform_Visualizer(symbol_duration=1.0, sampling_rate=1000, carrier_freq=2)
    
    print("QPSK调制波形可视化")
    print("=" * 50)
    
    # 方法1：详细视图（每个符号有波形和星座图）
    print("显示详细QPSK符号波形...")
    visualizer.plot_qpsk_symbols()
    
    # 方法2：简洁比较视图
    print("显示QPSK符号波形比较...")
    #visualizer.plot_comparison()
    
    # 方法3：带相位标注的视图
    print("显示带相位标注的QPSK波形...")
    #visualizer.plot_with_phase_annotation()
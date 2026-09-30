# -*- coding: utf-8 -*-
"""生成优化历程对比图（5 类子集，同一测试集口径）。"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs(r'D:\pytorch\anomalib_test\DINOv2AD_results', exist_ok=True)

configs = [
    'V0 baseline\n224/single\nmed+morph',
    'V1 +layers[7,9,11]\n224',
    'V2 +448res\nsingle',
    'V3 -uint8-post\n448/single',
    'V4 +2layers[7,9]\n448',
    'V5 +cat-routing\n448/2layers',
    'V6 +3layers[7,9,11]\n448/routing',
    'V7 +560res\n448→560',
]
f1 = [0.4183, 0.4264, 0.5433, 0.6026, 0.6359, 0.6392, 0.6428, 0.6647]
auroc = [0.6178, 0.6533, 0.6745, 0.9721, 0.9889, 0.9888, 0.9891, 0.9900]
ap = [0.5169, 0.5169, 0.5169, 0.5500, 0.5808, 0.5913, 0.6000, 0.6199]
latency = [13.8, 14.4, 63.1, 63.3, 96.5, 14.9, 14.9, 26.8]

fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
fig.suptitle('DINOv2AD Optimization History (5-category subset, same test set)', fontsize=14, fontweight='bold')

ax = axes[0]
x = range(len(configs))
ax.plot(x, f1, marker='o', linewidth=2, label='Pixel F1', color='#d62728')
ax.plot(x, auroc, marker='s', linewidth=2, label='Pixel AUROC', color='#1f77b4')
ax.plot(x, ap, marker='^', linewidth=2, label='Pixel AP', color='#2ca02c')
for i, (a, b, c) in enumerate(zip(f1, auroc, ap)):
    ax.annotate(f'{a:.3f}', (i, a), textcoords='offset points', xytext=(0, 8), fontsize=8, ha='center', color='#d62728')
    ax.annotate(f'{b:.3f}', (i, b), textcoords='offset points', xytext=(0, -14), fontsize=8, ha='center', color='#1f77b4')
    ax.annotate(f'{c:.3f}', (i, c), textcoords='offset points', xytext=(0, -26), fontsize=8, ha='center', color='#2ca02c')
ax.set_xticks(list(x))
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylabel('Score')
ax.set_ylim(0.35, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=10)
ax.set_title('Pixel-level metrics', fontsize=12)

ax = axes[1]
colors = ['#7f7f7f'] * 5 + ['#ff7f0e', '#2ca02c', '#d62728']
ax.bar(x, latency, color=colors, width=0.6)
for i, v in enumerate(latency):
    ax.annotate(f'{v:.1f}ms', (i, v), textcoords='offset points', xytext=(0, 3), fontsize=9, ha='center')
ax.axhline(100, color='red', linestyle='--', linewidth=1.5)
ax.text(7.25, 101, '100ms limit', color='red', fontsize=9, ha='right')
ax.set_xticks(list(x))
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylabel('Max inference latency (ms)')
ax.set_ylim(0, 115)
ax.grid(alpha=0.3)
ax.set_title('Inference latency (100ms hard limit)', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = r'D:\pytorch\anomalib_test\DINOv2AD_results\optimization_history.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved:', out)

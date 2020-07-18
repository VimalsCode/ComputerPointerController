import matplotlib.pyplot as plt
import numpy as np

"""
Helper to create chart
"""
labels = ['m_fd', 'm_hpe', 'm_fld', 'm_ge']
inference_time_FP32 = [0.268, 0.105, 0.084, 0.12902]
inference_time_FP16 = [0.27502, 0.129, 0.09785, 0.14403]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
inf_fp32 = ax.bar(x - width / 2, inference_time_FP32, width, label='FP32')
inf_fp16 = ax.bar(x + width / 2, inference_time_FP16, width, label='FP16')

ax.set_ylabel('Load time')
ax.set_title('Model load time by precision')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(inf_fp32)
autolabel(inf_fp16)

fig.tight_layout()

plt.show()
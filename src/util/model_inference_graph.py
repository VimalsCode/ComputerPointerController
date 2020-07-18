import matplotlib.pyplot as plt
import numpy as np

"""
Helper to create chart
"""
labels = ['m_fd', 'm_hpe', 'm_fld', 'm_ge']
inference_time_FP32 = [0.01186, 0.00141, 0.00115, 0.00175]
inference_time_FP16 = [0.01134, 0.00138, 0.00121, 0.00165]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
inf_fp32 = ax.bar(x - width / 2, inference_time_FP32, width, label='FP32')
inf_fp16 = ax.bar(x + width / 2, inference_time_FP16, width, label='FP16')

ax.set_ylabel('Inference time')
ax.set_title('Model inference by precision')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(inf_fp32)
autolabel(inf_fp16)

fig.tight_layout()

plt.show()
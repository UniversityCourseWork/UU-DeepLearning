import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab

tensorboard_summary_path = "./temp/summary/"

RUN_DATA = dict()
RUN_INDX = 0
for subdir, dirs, files in os.walk(tensorboard_summary_path):
    for file in files:

        RUN_INDX += 1
        RUN_DATA[f"RUN_{RUN_INDX}_TRLS"] = []
        RUN_DATA[f"RUN_{RUN_INDX}_VLLS"] = []
        RUN_DATA[f"RUN_{RUN_INDX}_TRPPL"] = []
        RUN_DATA[f"RUN_{RUN_INDX}_VLPPL"] = []
        
        for e in tf.compat.v1.train.summary_iterator(os.path.join(subdir, file)):
            if len(e.summary.value) == 1:
                if e.summary.value[0].tag == "Train_Loss":
                    if e.summary.value[0].simple_value < 7.0:
                        RUN_DATA[f"RUN_{RUN_INDX}_TRLS"].append(e.summary.value[0].simple_value)
                elif e.summary.value[0].tag == "Train_PPL":
                    if e.summary.value[0].simple_value < 800.0:
                        RUN_DATA[f"RUN_{RUN_INDX}_TRPPL"].append(e.summary.value[0].simple_value)
                elif e.summary.value[0].tag == "Valid_Loss":
                    if e.summary.value[0].simple_value < 6.75:
                        RUN_DATA[f"RUN_{RUN_INDX}_VLLS"].append(e.summary.value[0].simple_value)
                elif e.summary.value[0].tag == "Valid_PPL":
                    if e.summary.value[0].simple_value < 800.0:
                        RUN_DATA[f"RUN_{RUN_INDX}_VLPPL"].append(e.summary.value[0].simple_value)


# Plot the graph from extracted data
fig, axs = plt.subplots(2, 2, figsize=(9, 6))

for id in range(1, RUN_INDX+1):
    axs[0, 0].plot(RUN_DATA[f"RUN_{id}_TRLS"], label=f"Experiment {id}")
    axs[0, 1].plot(RUN_DATA[f"RUN_{id}_TRPPL"], label=f"Experiment {id}")
    axs[1, 0].plot(RUN_DATA[f"RUN_{id}_VLLS"], label=f"Experiment {id}")
    axs[1, 1].plot(RUN_DATA[f"RUN_{id}_VLPPL"], label=f"Experiment {id}")

axs[0, 0].set(ylabel="Training Loss")
axs[0, 1].set(ylabel="PPL")
axs[1, 0].set(ylabel="Validation Loss", xlabel="Epoch")
axs[1, 1].set(ylabel="PPL", xlabel="Epoch")


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
labels = list(set(labels))
labels.sort(key=lambda x: int(x.split()[-1]))
fig.legend(lines, labels, bbox_to_anchor=(1.10, 0.5), loc="center right")
plt.savefig("./temp/graphs/Graph.pdf", format="pdf", bbox_inches="tight")



#plt.figure()
# for id in range(1, RUN_INDX+1):
#     plt.plot(RUN_DATA[f"RUN_{id}_TRLS"], label=f"Experiment {id}")
# plt.xlabel("Epoch")
# plt.ylabel("Training Loss")
# plt.legend(ncols=1, bbox_to_anchor=(1.00, 1.00), loc="upper left")
# plt.savefig("./temp/graphs/Train_Loss.pdf", format="pdf", bbox_inches="tight")

# plt.figure()
# for id in range(1, RUN_INDX+1):
#     plt.plot(RUN_DATA[f"RUN_{id}_TRPPL"], label=f"Experiment {id}")
# plt.xlabel("Epoch")
# plt.ylabel("Training Perpelexity")
# #plt.legend(ncols=2, bbox_to_anchor=(1.00, 1.00), loc="upper left")
# plt.savefig("./temp/graphs/Train_PPL.pdf", format="pdf", bbox_inches="tight")

# plt.figure()
# for id in range(1, RUN_INDX+1):
#     plt.plot(RUN_DATA[f"RUN_{id}_VLLS"], label=f"Experiment {id}")
# plt.xlabel("Epoch")
# plt.ylabel("Validation Loss")
# #plt.legend(ncols=2, bbox_to_anchor=(1.00, 1.00), loc="upper left")
# plt.savefig("./temp/graphs/Valid_Loss.pdf", format="pdf", bbox_inches="tight")

# plt.figure()
# for id in range(1, RUN_INDX+1):
#     plt.plot(RUN_DATA[f"RUN_{id}_VLPPL"], label=f"Experiment {id}")
# plt.xlabel("Epoch")
# plt.ylabel("Validation Perpelexity")
# plt.legend(ncols=2, bbox_to_anchor=(1.00, 1.00), loc="upper left")
# plt.savefig("./temp/graphs/Valid_PPL.pdf", format="pdf", bbox_inches="tight")


# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)
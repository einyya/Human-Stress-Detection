import bioread
import pandas as pd
import matplotlib.pyplot as plt

file_path=("D:\Participants\music_group\P_10\P_10_.acq")
task= bioread.read_file(file_path)

# plt.plot(task.channels[0].data[20000:25000])
# plt.show()
# plt.plot(task.channels[1].data[20000:25000])
# plt.show()
# plt.plot(task.channels[2].data[20000:25000])
# plt.show()
# data1=pd.DataFrame(data1)
# data1.to_csv('data1.csv')
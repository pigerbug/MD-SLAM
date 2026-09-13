import matplotlib.pyplot as plt


# 读取文件数据

data = []

total_data = []

data_10=[]
with open("EpiErr.txt", "r") as file:
    for line in file:
        tt = float(line.strip())
        total_data.append(tt)
        if(tt<0.5):
            data.append(tt)
        if(tt<5):
            data_10.append(tt)

# 绘制直方图
plt.hist(data_10, bins=10, edgecolor='black')

# 添加标题和标签
plt.title('Histogram of Data from data.txt')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 计算占比和平均值
out = "总数据size: {}, value个数: {}, 占比: {:.2%}, data_10: {}, 占比: {:.2%}".format(len(total_data), len(data), len(data)/len(total_data), len(data_10), len(data_10)/len(total_data))
print(out)

# 显示图表
plt.show()

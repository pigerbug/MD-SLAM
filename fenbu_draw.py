import matplotlib.pyplot as plt

value = 0
# 读取文件数据
data = []
sum = []
with open("fenbu.txt", "r") as file:
    for line in file:
        tt = float(line.strip())
        sum.append(tt)
        if(tt>0.96):
            value = value+1
            data.append(tt)

# 绘制直方图
plt.hist(sum, bins=10, edgecolor='black')

# 添加标题和标签
plt.title('Histogram of Data from data.txt')
plt.xlabel('Value')
plt.ylabel('Frequency')
out = "总数据size: {}, value个数: {}, 占比: {}".format(len(sum), value, value/len(sum))
print(out)

# 显示图表
plt.show()

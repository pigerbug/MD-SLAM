import matplotlib.pyplot as plt

value = 0
less_8 = 0
# 读取文件数据
data = []
sum = []
with open("cal_f_num.txt", "r") as file:
    for line in file:
        tt = int(line.strip())
        sum.append(tt)
        if(tt<8):
            less_8 +=1
        if(tt>15):
            value += 1
        
        

# 绘制直方图
plt.hist(sum, bins=10, edgecolor='black')

# 添加标题和标签
plt.title('Histogram of Data from data.txt')
plt.xlabel('Value')
plt.ylabel('Frequency')
out = "总数据size: {}, value个数: {}, 占比: {}, more8占比: {}".format(len(sum), value, value/len(sum), less_8)
print(out)

# 显示图表
plt.show()

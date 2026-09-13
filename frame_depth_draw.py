import matplotlib.pyplot as plt
import argparse

def main():

    parser = argparse.ArgumentParser(description="Process a file and plot histogram.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("thread", help="thread", type=float)
    
    args = parser.parse_args()
    
    thread = args.thread
    input_file = args.input_file

    data = []
    total_data = []
    aa = 0
    max_value = 0
    with open(input_file, "r") as file:
        for line in file:
            tt = float(line.strip())
            total_data.append(tt)
            max_value = max(max_value, tt)
            aa += tt
            if tt < thread:
                data.append(tt)

    # 绘制直方图
    plt.hist(data, bins=10, edgecolor='black')

    # 添加标题和标签
    plt.title('Histogram of Data from {}'.format(input_file))
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 计算占比和平均值
    out = "阈值: {}, 总数据size: {}, value个数: {}, 占比: {:.2%}, mean: {}, 最大值：{}".format(thread, len(total_data), len(data), len(data)/len(total_data), aa/len(total_data), max_value)
    print(out)

    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()

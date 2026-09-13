import matplotlib.pyplot as plt
import argparse
import numpy as np


def main():

    parser = argparse.ArgumentParser(description="Process a file and plot histogram.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("mvalue", help="数据筛选的最大值", type=float, nargs="?", default=None)
    parser.add_argument("thread", help="数据筛选的阈值", type=float, nargs="?", default=None)
    args = parser.parse_args()
    
    thread = args.thread
    input_file = args.input_file
    mvalue = args.mvalue
    data = []
    total_data = []
    aa = 0
    max_value = 0
    with open(input_file, "r") as file:
        for line in file:
            tt = float(line.strip())
            if(tt<mvalue):
                total_data.append(tt)
                max_value = max(max_value, tt)
                aa += tt
                if thread is None or tt < thread:  # 如果没有给出thread，则不筛选数据
                    data.append(tt)
            
    mean = np.mean(total_data)
    stddev = np.std(total_data)
    simulated_data = np.random.normal(loc=mean, scale=stddev, size=len(total_data))
    # 绘制原始数据和模拟的正态分布数据的直方图
    plt.hist(total_data, bins=10, alpha=0.5, label="raw", edgecolor='black')
    plt.hist(simulated_data, bins=10, alpha=0.5, label="zhengtai", edgecolor='red')

    simulated_data = np.random.normal(loc=mean, scale=stddev, size=len(total_data))

  



    # 添加标题和标签
    plt.title('raw with zhengtai')
    plt.xlabel('value')
    plt.ylabel('f')
    plt.legend()

    # 输出统计信息
    aa = np.sum(total_data)
    max_value = np.max(total_data)
    out = "数据总量: {}, 正态分布均值: {:.2f}, 正态分布标准差: {:.2f}, 最大值: {:.2f}, 截断占比: {:.2f}, 截断的数据量: {}".format(
        len(total_data), mean, stddev, max_value, len(data)/len(total_data), len(data)
    )
    print(out)

    # 显示图表
    plt.show()

   

if __name__ == "__main__":
    main()

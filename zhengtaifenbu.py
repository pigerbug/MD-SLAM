import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="读取文件数据并生成正态分布数据。")
    parser.add_argument("input_file", help="输入文件路径")
    parser.add_argument("thread", help="数据筛选的阈值", type=float, nargs="?", default=None)
    
    args = parser.parse_args()
    
    input_file = args.input_file
    thread = args.thread

    # 从文件读取数据
    data = []
    total_data = []
    with open(input_file, "r") as file:
        for line in file:
            try:
                value = float(line.strip())
                data.append(value)
                total_data.append(value)
            except ValueError:
                continue  # 忽略不能转换为数字的行

    # 计算数据的均值和标准差
    mean = np.mean(data)
    stddev = np.std(data)
    print(mean,stddev)
    # 生成符合正态分布的数据
    simulated_data = np.random.normal(loc=mean, scale=stddev, size=len(data))

    # 根据阈值筛选数据
    if thread is not None:
        filtered_data = [x for x in simulated_data if x < thread]
    else:
        filtered_data = simulated_data

    # 绘制原始数据和模拟的正态分布数据的直方图
    plt.hist(data, bins=10, alpha=0.5, label="原始数据", edgecolor='black')
    plt.hist(filtered_data, bins=10, alpha=0.5, label="模拟的正态分布数据", edgecolor='red')

    # 添加标题和标签
    plt.title('原始数据与模拟正态分布数据的对比')
    plt.xlabel('值')
    plt.ylabel('频率')
    plt.legend()

    # 输出统计信息
    aa = np.sum(data)
    max_value = np.max(data)
    out = "数据总量: {}, 正态分布均值: {:.2f}, 正态分布标准差: {:.2f}, 最大值: {:.2f}".format(
        len(total_data), mean, stddev, max_value
    )
    print(out)

    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()

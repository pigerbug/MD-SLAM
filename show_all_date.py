import matplotlib.pyplot as plt
import argparse

def main():

    parser = argparse.ArgumentParser(description="Process a file and plot histogram.")
    parser.add_argument("input_file", help="Path to the input file")
    #parser.add_argument("thread", help="thread", type=float)
    
    args = parser.parse_args()
    
    #thread = args.thread
    input_file = args.input_file

    data = []
    total_data = []
    aa = 0
    value_max = 0.0
    with open(input_file, "r") as file:
        for line in file:
            tt = float(line.strip())
            total_data.append(tt)
            aa += tt
            value_max = max(value_max, tt)

    # 绘制直方图
    plt.hist(total_data, bins=10, edgecolor='black')

    # 添加标题和标签
    plt.title('Histogram of Data from {}'.format(input_file))
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 计算占比和平均值
    out = " 总数据size: {}, value个数: {}, 占比: {:.2%}, mean: {}, 最大值: {}".format( len(total_data), len(data), len(data)/len(total_data), aa/len(total_data), value_max)
    print(out)
    plt.savefig('yolo_depth.png')
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()

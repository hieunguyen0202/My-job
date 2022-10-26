
import numpy as np
import matplotlib.pyplot as plt


# Mean Squared Error là tổng giá trị bình phương của hiệu giữa actual và predicted values.
# xây dựng hàm tính sai số bình phương
def mean_squared_error(y_true, y_predicted):
    # Tính toán hàm loss và hàm cost
    #cost = (giá trị actual - giá trị predict)^2 / size(y_actual)
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost


# Thực hiện phương pháp Gradient Descent Function
# số vòng lặp (iterations),tỉ lệ học (learning_rate),ngưỡng dùng để xác định điều kiện dùng vòng lặp (stopping_threshold)
# Xây dựng hàm gradient_descent với các thông số trên có thể điều chỉnh (1000 vong lặp, learning-rate=0.0001, ngưỡng dừng 0.0001)
def gradient_descent(x, y, iterations=1000, learning_rate=0.00000001,
                     stopping_threshold=1e-6):
    # Khởi tạo weight, bias, learning rate và iterations
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))  #tổng số phần tử X

    costs = []
    weights = []
    previous_cost = None

    # khởi tạo vòng lặp
    for i in range(iterations):

        # Giá trị dự đoán  ( (y) = (w * x) + b)
        y_predicted = (current_weight * x) + current_bias

        # Tính toán current cost
        current_cost = mean_squared_error(y, y_predicted)

        # Nếu sự thay đổi của hàm cost ít hơn or bằng gái trị stopping_threshold thì ta dừng vòng lặp
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break

        previous_cost = current_cost

        costs.append(current_cost)
        weights.append(current_weight)

        # Tính toán các giá trị đạo hàm gradients
        weight_derivative = -(2 / n) * sum(x * (y - y_predicted))
        bias_derivative = -(2 / n) * sum(y - y_predicted)

        # Cập nhật giá trị weights và bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

        # In các thong số ra màn hình sau 1000 bước lặp
        print(f"Iteration {i + 1}: Cost {current_cost}, Weight \
        {current_weight}, Bias {current_bias}")

    # hiển thị các thông số weights và cost cho tất cả vòng lặp
    plt.figure(figsize=(8, 6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()

    return current_weight, current_bias


def main():
    # Khởi tạo Data random
    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
                  55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
                  45.41973014, 54.35163488, 44.1640495, 58.16847072, 56.72720806,
                  48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    Y = np.array([31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513,
                  78.21151827, 79.64197305, 59.17148932, 75.3312423, 71.30087989,
                  55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
                  60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
    # X = np.array([23.50234527, 55.42680403,12.53035803, 78.47563963, 34.81320787,
    #               55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
    #               67.41973014, 23.35163488,35.1640495, 6.16847072, 88.72720806,
    #               33.95588857, 20.68719623, 60.29732685, 12.61864377, 78.81681754])
    # Y = np.array([31.70700585, 68.77759598, 62.5623823, 78.54663223, 87.23092513,
    #               89.21151827, 22.64197305, 25.17148932, 16.3312423, 71.30087989,
    #               55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
    #               67.72360244, 82.89250373, 45.37989686, 26.84715332, 56.87721319])

    # Ước tính weight and bias sử dụng gradient descent
    estimated_weight, eatimated_bias = gradient_descent(X, Y, iterations=2000)
    print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {eatimated_bias}")

    # Tính toán giá trị dự đoán dựa trên các thông số đã ước tính
    Y_pred = estimated_weight * X + eatimated_bias

    # Vẽ đường line tốt nhất  (dùng linear regression)
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, marker='o', color='red')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='green', markerfacecolor='red',
             markersize=10, linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == "__main__":
    main()
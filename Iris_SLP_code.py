import math
import matplotlib.pyplot as plt

#Iris dataset
IRIS = [
    (5.1,3.5,1.4,0.2,0),(4.9,3.0,1.4,0.2,0),(4.7,3.2,1.3,0.2,0),(4.6,3.1,1.5,0.2,0),
    (5.0,3.6,1.4,0.2,0),(5.4,3.9,1.7,0.4,0),(4.6,3.4,1.4,0.3,0),(5.0,3.4,1.5,0.2,0),
    (4.4,2.9,1.4,0.2,0),(4.9,3.1,1.5,0.1,0),(5.4,3.7,1.5,0.2,0),(4.8,3.4,1.6,0.2,0),
    (4.8,3.0,1.4,0.1,0),(4.3,3.0,1.1,0.1,0),(5.8,4.0,1.2,0.2,0),(5.7,4.4,1.5,0.4,0),
    (5.4,3.9,1.3,0.4,0),(5.1,3.5,1.4,0.3,0),(5.7,3.8,1.7,0.3,0),(5.1,3.8,1.5,0.3,0),
    (5.4,3.4,1.7,0.2,0),(5.1,3.7,1.5,0.4,0),(4.6,3.6,1.0,0.2,0),(5.1,3.3,1.7,0.5,0),
    (4.8,3.4,1.9,0.2,0),(5.0,3.0,1.6,0.2,0),(5.0,3.4,1.6,0.4,0),(5.2,3.5,1.5,0.2,0),
    (5.2,3.4,1.4,0.2,0),(4.7,3.2,1.6,0.2,0),(4.8,3.1,1.6,0.2,0),(5.4,3.4,1.5,0.4,0),
    (5.2,4.1,1.5,0.1,0),(5.5,4.2,1.4,0.2,0),(4.9,3.1,1.5,0.1,0),(5.0,3.2,1.2,0.2,0),
    (5.5,3.5,1.3,0.2,0),(4.9,3.1,1.5,0.1,0),(4.4,3.0,1.3,0.2,0),(5.1,3.4,1.5,0.2,0),
    (5.0,3.5,1.3,0.3,0),(4.5,2.3,1.3,0.3,0),(4.4,3.2,1.3,0.2,0),(5.0,3.5,1.6,0.6,0),
    (5.1,3.8,1.9,0.4,0),(4.8,3.0,1.4,0.3,0),(5.1,3.8,1.6,0.2,0),(4.6,3.2,1.4,0.2,0),
    (5.3,3.7,1.5,0.2,0),(5.0,3.3,1.4,0.2,0),
    (7.0,3.2,4.7,1.4,1),(6.4,3.2,4.5,1.5,1),(6.9,3.1,4.9,1.5,1),(5.5,2.3,4.0,1.3,1),
    (6.5,2.8,4.6,1.5,1),(5.7,2.8,4.5,1.3,1),(6.3,3.3,4.7,1.6,1),(4.9,2.4,3.3,1.0,1),
    (6.6,2.9,4.6,1.3,1),(5.2,2.7,3.9,1.4,1),(5.0,2.0,3.5,1.0,1),(5.9,3.0,4.2,1.5,1),
    (6.0,2.2,4.0,1.0,1),(6.1,2.9,4.7,1.4,1),(5.6,2.9,3.6,1.3,1),(6.7,3.1,4.4,1.4,1),
    (5.6,3.0,4.5,1.5,1),(5.8,2.7,4.1,1.0,1),(6.2,2.2,4.5,1.5,1),(5.6,2.5,3.9,1.1,1),
    (5.9,3.2,4.8,1.8,1),(6.1,2.8,4.0,1.3,1),(6.3,2.5,4.9,1.5,1),(6.1,2.8,4.7,1.2,1),
    (6.4,2.9,4.3,1.3,1),(6.6,3.0,4.4,1.4,1),(6.8,2.8,4.8,1.4,1),(6.7,3.0,5.0,1.7,1),
    (6.0,2.9,4.5,1.5,1),(5.7,2.6,3.5,1.0,1),(5.5,2.4,3.8,1.1,1),(5.5,2.4,3.7,1.0,1),
    (5.8,2.7,3.9,1.2,1),(6.0,2.7,5.1,1.6,1),(5.4,3.0,4.5,1.5,1),(6.0,3.4,4.5,1.6,1),
    (6.7,3.1,4.7,1.5,1),(6.3,2.3,4.4,1.3,1),(5.6,3.0,4.1,1.3,1),(5.5,2.5,4.0,1.3,1),
    (5.5,2.6,4.4,1.2,1),(6.1,3.0,4.6,1.4,1),(5.8,2.6,4.0,1.2,1),(5.0,2.3,3.3,1.0,1),
    (5.6,2.7,4.2,1.3,1),(5.7,3.0,4.2,1.2,1),(5.7,2.9,4.2,1.3,1),(6.2,2.9,4.3,1.3,1),
    (5.1,2.5,3.0,1.1,1),(5.7,2.8,4.1,1.3,1),
]

#split dataset
train_data = IRIS[0:40] + IRIS[50:90]
val_data   = IRIS[40:50] + IRIS[90:100]

LEARNING_RATE = 0.1
EPOCHS = 5

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def forward(x, w):
    bias, t1, t2, t3, t4 = w
    z = bias + t1 * x[0] + t2 * x[1] + t3 * x[2] + t4 * x[3]
    return sigmoid(z)

def evaluate(dataset, w):
    total_loss = 0.0
    correct = 0
    for x1, x2, x3, x4, target in dataset:
        o = forward((x1, x2, x3, x4), w)
        pred = 1 if o > 0.5 else 0
        error = o - target
        total_loss += error ** 2
        if pred == target:
            correct += 1
    n = len(dataset)
    return total_loss / n, correct / n

def train_one_epoch(dataset, w):
    bias, t1, t2, t3, t4 = w
    total_loss = 0.0
    correct = 0
    for x1, x2, x3, x4, target in dataset:
        o = forward((x1, x2, x3, x4), (bias, t1, t2, t3, t4))
        pred = 1 if o > 0.5 else 0
        error = o - target
        total_loss += error ** 2
        if pred == target:
            correct += 1
        d_bias = 2 * error * (1 - o) * o
        d_t1 = d_bias * x1
        d_t2 = d_bias * x2
        d_t3 = d_bias * x3
        d_t4 = d_bias * x4

        bias -= LEARNING_RATE * d_bias
        t1 -= LEARNING_RATE * d_t1
        t2 -= LEARNING_RATE * d_t2
        t3 -= LEARNING_RATE * d_t3
        t4 -= LEARNING_RATE * d_t4
    n = len(dataset)
    return (bias, t1, t2, t3, t4), total_loss / n, correct / n

def main():
    w = (0.5, 0.5, 0.5, 0.5, 0.5)  # bias, teta1..teta4

    train_loss_hist, train_acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []

    for epoch in range(1, EPOCHS + 1):
        w, tr_loss, tr_acc = train_one_epoch(train_data, w)
        val_loss, val_acc = evaluate(val_data, w)

        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        print(f"Epoch {epoch}: train_loss={tr_loss:.6f} train_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}")

    print("\nFinal weights (bias, teta1, teta2, teta3, teta4):")
    print(w)

    plot_results(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist)

    return {
        "train_loss": train_loss_hist,
        "train_acc": train_acc_hist,
        "val_loss": val_loss_hist,
        "val_acc": val_acc_hist,
    }

def plot_results(train_loss, val_loss, train_acc, val_acc):
    epochs = list(range(1, len(train_loss) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def single_chart(ax, values, title, ylabel, color):
        ax.plot(epochs, values, marker='o', color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)

    single_chart(axes[0, 0], train_loss, 'Training Loss per Epoch', 'Loss (avg SSE)', '#A5D6A7')
    single_chart(axes[0, 1], train_acc, 'Training Accuracy per Epoch', 'Accuracy', '#092328')
    single_chart(axes[1, 0], val_loss, 'Validation Loss per Epoch', 'Loss (avg SSE)', '#800020')
    single_chart(axes[1, 1], val_acc, 'Validation Accuracy per Epoch', 'Accuracy', '#B153D7')

    plt.tight_layout()
    plt.savefig('slp_charts.png', dpi=150)
    print("Chart saved as slp_charts.png")
    plt.show()

if __name__ == "__main__":
    main()
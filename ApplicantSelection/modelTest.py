import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from createDataBase import generate_hiring_data

def svm_classification_with_plot(return_metrics=True, show_plot=False):

    df = generate_hiring_data()

    X = df[["experience_years", "technical_score"]]
    y = df["hire_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = SVC(kernel="linear", C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()  # numpy array JSON'a çevrilemez
    report = classification_report(y_test, y_pred, output_dict=True)  # dict formatta döner

    if show_plot:
        plot_decision_boundary(model, X_scaled, y)

    if return_metrics:
        return {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": report
        }

def plot_decision_boundary(model, X, y):
    h = 0.02  # grid adımı
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    plt.title("SVM Karar Sınırı (Linear Kernel)")
    plt.xlabel("Experience (scaled)")
    plt.ylabel("Technical Score (scaled)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    svm_classification_with_plot()

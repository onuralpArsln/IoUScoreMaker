import numpy as np
import matplotlib.pyplot as plt

def graph(data1:list,data2:list,graphName:str,xname:str = "Kernel Size",yname:str = "iou score"):
    x = np.array(data1)
    y = np.array(data2)
    p = np.polyfit(x, y, deg=3)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = np.polyval(p, x_fit)
        # Plot original data points
    plt.scatter(x, y, label='Data points')

    # Plot fitted curve
    plt.plot(x_fit, y_fit, 'r', label='Fitted curve')

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(graphName)

    # Show legend
    plt.legend()

    plt.savefig(graphName+'.jpg', format='jpg')
    # Show plot
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Sample data (replace these with your data)
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1])

    # Fit a polynomial curve
    p = np.polyfit(x, y, deg=3)  # Change deg to change the degree of the polynomial

    # Generate points for the fitted curve
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = np.polyval(p, x_fit)

    # Plot original data points
    plt.scatter(x, y, label='Data points')

    # Plot fitted curve
    plt.plot(x_fit, y_fit, 'r', label='Fitted curve')

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Curve Fitting')

    # Show legend
    plt.legend()



    plt.savefig('curve_fit_test.jpg', format='jpg')
    # Show plot
    plt.grid(True)
    plt.show()

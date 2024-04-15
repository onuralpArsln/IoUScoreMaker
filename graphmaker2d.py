import numpy as np
import matplotlib.pyplot as plt

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

import numpy as np
import matplotlib.pyplot as plt




def graph(kernelSize:list,iouScore:list,graphName:str,xname:str = "Kernel Size",yname:str = "iou score"):
    """
    Bu fonksiyon verilen data noktaları ile grafik oluşturur ve curve fitting uygular
    
    Parameters:
    kernelSize (list): x eksenine gelecek datalar arrayı, başta kernel size dğiştim ondan adı bu.

    iouScore (list): y eksenine gelecek datalar arrayı, buraya genelde iou score koyulacak

    graphName (string) : grafik adı aynı zamanda sonuç bu isim ile jpg olarak kaydedilecek

    xname (string): x ekseni ismi

    yname (string): y ekseni ismi
    
    Returns:
    void,  bir jpgyi direkt kaydeder
    """
    x = np.array(kernelSize)
    y = np.array(iouScore)
    p = np.polyfit(x, y, deg=3)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = np.polyval(p, x_fit)
        # Plot original data points
    plt.scatter(x, y, label='Data points')

    # Plot fitted curve
    plt.plot(x_fit, y_fit, 'r', label='Fitted curve')

    # Set labels and title
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(graphName)

    # Show legend
    plt.legend()

    plt.savefig(graphName+'.jpg', format='jpg')
    # Show plot
    plt.grid(True)
    plt.show()





##bu alt test için 
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

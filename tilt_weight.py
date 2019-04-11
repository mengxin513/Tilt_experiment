from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with PdfPages('tilt_weight.pdf') as pdf:

        torque = np.array([-0.01, -0.005, -0.004, -0.002, -0.001, 0, 0.001, 0.002, 0.004, 0.005, 0.01])
        pixel_shift = np.array([-23.89, -11.58, -9.20, -4.55, -2.29, 0.49, 1.72, 3.95, 8.53, 10.44, 21.13])

        angle_per_pixel = 0.0586

        angular_shift = pixel_shift * angle_per_pixel

        matplotlib.rcParams.update({'font.size': 12})

        fig, ax = plt.subplots(1, 1)

        ax.plot(torque, angular_shift, 'ro')

        coef = np.polyfit(torque, angular_shift, 2)
        print(r'y = {}x^2 + {}x + {}'.format(coef[0], coef[1], coef[2]))
        xp = np.linspace(1.1 * min(torque), 1.1 * max(torque), 100)
        p = np.poly1d(coef)
        ax.plot(xp, p(xp), 'b-')

        xmin, xmax = ax.get_xlim()
        r = max(abs(xmin), abs(xmax))
        ax.set_xlim(-r, r)

        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        plt.xlabel(r'Torque [$\mathrm{kgm}$]', horizontalalignment = 'right', x = 1.0)
        plt.ylabel(r'Angular shift [$\mathrm{mrad}$]', horizontalalignment = 'right', y = 1.0)

        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

        regression = angular_shift - p(torque)

        fig, ax = plt.subplots(1, 1)

        ax.plot(torque, regression, 'ro-')

        xmin, xmax = ax.get_xlim()
        r = max(abs(xmin), abs(xmax))
        ax.set_xlim(-r, r)

        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        plt.xlabel(r'Torque [$\mathrm{kgm}$]', horizontalalignment = 'right', x = 1.0)
        plt.ylabel(r'Regression [$\mathrm{mrad}$]', horizontalalignment = 'right', y = 1.0)

        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

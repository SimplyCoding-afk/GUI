import numpy as np
import matplotlib.pyplot as plt

# -------- Given Values (Converted to nm) --------
A = 100        # nm
B = 11700       # nm^2/hr
t_input = 1    # hr

x_exp_nm = 36.8   # SEM measured thickness (nm)

# -------- Deal-Grove Thickness --------
x_theory_nm = (-A + np.sqrt(A**2 + 4*B*t_input)) / 2

# -------- Error --------
error = abs(x_exp_nm - x_theory_nm) / x_exp_nm * 100

# -------- Print Results --------
print("Measured Thickness (SEM) :", x_exp_nm, "nm")
print("Deal–Grove Prediction    :", round(x_theory_nm,2), "nm")
print("Oxidation Type           : Dry Oxidation")
print("Temperature              : 1000 °C")
print("Oxidation Time           :", t_input, "hour")
print("Error                    :", round(error,2), "%")

# -------- Graph --------
t = np.linspace(0, 2, 200)

x_curve_nm = (-A + np.sqrt(A**2 + 4*B*t)) / 2

plt.plot(t, x_curve_nm, label="Deal-Grove Model")
plt.scatter(t_input, x_theory_nm, label="Prediction at 1 hr")
plt.scatter(t_input, x_exp_nm, label="SEM Measurement")

plt.xlabel("Time (hours)")
plt.ylabel("Oxide Thickness (nm)")
plt.title("Oxide Growth vs Time (Dry Oxidation, 1000°C)")
plt.legend()
plt.show()

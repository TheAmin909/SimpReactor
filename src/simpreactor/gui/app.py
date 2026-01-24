"""Qt application entrypoint for the SimpReactor GUI."""

from __future__ import annotations

import sys

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from simpreactor.gui.simulation import CSTRInputs, run_cstr_simulation


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(1, 1, 1)

    def plot_result(self, result) -> None:
        self.axes.clear()
        self.axes.plot(result.time, result.a, label="A")
        self.axes.plot(result.time, result.b, label="B")
        self.axes.plot(result.time, result.temperature, label="T")
        self.axes.plot(result.time, result.jacket_temperature, label="Tj")
        self.axes.set_xlabel("Time (s)")
        self.axes.legend()
        self.draw()


class CSTRWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SimpReactor CSTR")
        self.resize(1100, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        form_panel = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(form_panel)
        form_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.inputs = {
            "inflow_a": self._make_spin(1.5, 0.0, 10.0, 3),
            "inflow_b": self._make_spin(0.0, 0.0, 10.0, 3),
            "inflow_temperature": self._make_spin(350.0, 250.0, 800.0, 1),
            "inflow_rate": self._make_spin(0.15, 0.0, 5.0, 3),
            "volume": self._make_spin(1.0, 0.1, 10.0, 2),
            "density": self._make_spin(1000.0, 100.0, 2000.0, 1),
            "heat_capacity": self._make_spin(4200.0, 1000.0, 8000.0, 1),
            "ua": self._make_spin(1800.0, 0.0, 10000.0, 1),
            "jacket_mass": self._make_spin(45.0, 1.0, 500.0, 1),
            "jacket_heat_capacity": self._make_spin(4200.0, 1000.0, 8000.0, 1),
            "reaction_enthalpy": self._make_spin(-75000.0, -200000.0, 0.0, 1),
            "jacket_inlet_temperature": self._make_spin(320.0, 250.0, 800.0, 1),
            "jacket_heat_input": self._make_spin(0.0, -50000.0, 50000.0, 1),
            "pre_exponential": self._make_spin(2400.0, 0.0, 1e6, 1),
            "activation_energy": self._make_spin(85000.0, 1000.0, 200000.0, 1),
            "exponent_a": self._make_spin(1.0, 0.0, 3.0, 2),
            "duration": self._make_spin(200.0, 10.0, 2000.0, 1),
            "points": self._make_spin(200.0, 20.0, 2000.0, 0),
        }

        labels = {
            "inflow_a": "Inflow A (mol/L)",
            "inflow_b": "Inflow B (mol/L)",
            "inflow_temperature": "Inflow T (K)",
            "inflow_rate": "Flow rate (1/s)",
            "volume": "Volume (m³)",
            "density": "Density (kg/m³)",
            "heat_capacity": "Cp (J/kg/K)",
            "ua": "UA (W/K)",
            "jacket_mass": "Jacket mass (kg)",
            "jacket_heat_capacity": "Jacket Cp (J/kg/K)",
            "reaction_enthalpy": "ΔH (J/mol)",
            "jacket_inlet_temperature": "Jacket inlet T (K)",
            "jacket_heat_input": "Jacket heat input (W)",
            "pre_exponential": "Pre-exponential",
            "activation_energy": "Activation energy (J/mol)",
            "exponent_a": "Exponent A",
            "duration": "Duration (s)",
            "points": "Output points",
        }

        for key, widget in self.inputs.items():
            form_layout.addRow(labels[key], widget)

        self.run_button = QtWidgets.QPushButton("Run Simulation")
        self.run_button.clicked.connect(self._run_simulation)
        form_layout.addRow(self.run_button)

        self.plot_canvas = PlotCanvas()

        layout.addWidget(form_panel, stretch=1)
        layout.addWidget(self.plot_canvas, stretch=2)

    def _make_spin(self, value: float, minimum: float, maximum: float, decimals: int) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(10 ** (-decimals))
        return spin

    def _run_simulation(self) -> None:
        inputs = CSTRInputs(
            inflow_a=self.inputs["inflow_a"].value(),
            inflow_b=self.inputs["inflow_b"].value(),
            inflow_temperature=self.inputs["inflow_temperature"].value(),
            inflow_rate=self.inputs["inflow_rate"].value(),
            volume=self.inputs["volume"].value(),
            density=self.inputs["density"].value(),
            heat_capacity=self.inputs["heat_capacity"].value(),
            ua=self.inputs["ua"].value(),
            jacket_mass=self.inputs["jacket_mass"].value(),
            jacket_heat_capacity=self.inputs["jacket_heat_capacity"].value(),
            reaction_enthalpy=self.inputs["reaction_enthalpy"].value(),
            jacket_inlet_temperature=self.inputs["jacket_inlet_temperature"].value(),
            jacket_heat_input=self.inputs["jacket_heat_input"].value(),
            pre_exponential=self.inputs["pre_exponential"].value(),
            activation_energy=self.inputs["activation_energy"].value(),
            exponent_a=self.inputs["exponent_a"].value(),
            duration=self.inputs["duration"].value(),
            points=int(self.inputs["points"].value()),
        )

        result = run_cstr_simulation(inputs)
        self.plot_canvas.plot_result(result)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = CSTRWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

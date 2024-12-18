import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QMessageBox,
    QProgressBar,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QSizePolicy
import threading
import time


class ApplicationWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("PyQt6 Application")
        self.setGeometry(100, 100, 400, 300)

        # Layout
        layout = QVBoxLayout()

        # Logo Image at the Top
        self.logo_label = QLabel(self)
        self.logo_pixmap = QPixmap(
            "logo.png"
        )  # Ensure 'logo.png' is in the same directory or provide full path
        self.logo_label.setPixmap(self.logo_pixmap)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet("background-color: white;")
        layout.addWidget(self.logo_label)

        # Model Directory Button
        self.model_directory_button = QPushButton("Select Model File")
        self.model_directory_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.model_directory_button.clicked.connect(self.pick_model_file)
        layout.addWidget(self.model_directory_button)

        # Input Directory Button
        self.input_directory_button = QPushButton("Select Input Directory")
        self.input_directory_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.input_directory_button.clicked.connect(self.pick_input_directory)
        layout.addWidget(self.input_directory_button)

        # Export Directory Button
        self.export_directory_button = QPushButton("Select Export Directory")
        self.export_directory_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.export_directory_button.clicked.connect(self.pick_export_directory)
        layout.addWidget(self.export_directory_button)

        # Device Selection (CPU / GPU)
        self.device_label = QLabel("Select Device:")
        self.device_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.device_label.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.device_label)

        self.device_select = QComboBox()
        self.device_select.addItem("CPU")
        self.device_select.addItem("GPU")
        self.device_select.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.device_select)

        # Static Progress Bar (before the button)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(50)  # Static value for demonstration
        self.progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.progress_bar)

        # Show Alert Button
        self.show_alert_button = QPushButton("Show Alert")
        self.show_alert_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.show_alert_button.clicked.connect(self.show_alert)
        layout.addWidget(self.show_alert_button)

        # Set the layout
        self.setLayout(layout)

    def pick_model_file(self):
        file = QFileDialog.getOpenFileName(self, "Select Model")
        if file:
            self.input_directory_button.setText(f"Input: {file}")
        else:
            self.input_directory_button.setText("Select Model")

    def pick_input_directory(self):
        # Open a file dialog to pick the input directory
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")

        # If a directory is selected, update the button text
        if directory:
            self.input_directory_button.setText(f"Input: {directory}")
        else:
            self.input_directory_button.setText("Select Input Directory")

    def pick_export_directory(self):
        # Open a file dialog to pick the export directory
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")

        # If a directory is selected, update the button text
        if directory:
            self.export_directory_button.setText(f"Export: {directory}")
        else:
            self.export_directory_button.setText("Select Export Directory")

    def show_alert(self):
        def run():
            for i in range(100):
                self.progress_bar.setValue(i)
                time.sleep(0.1)

        QMessageBox.information(self, "Alert", "This is a static message!")

        thread = threading.Thread(target=run)
        thread.start()


# Main function to run the application
def main():
    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

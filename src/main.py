import sys
import os
import json
import pathlib
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QFrame,
    QVBoxLayout,
    QListWidgetItem,
    QMessageBox,
)
from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QIcon
from qfluentwidgets import (
    NavigationItemPosition,
    FluentWindow,
    TitleLabel,
    SubtitleLabel,
    setFont,
    FluentIcon as FIF,
    PrimaryPushButton,
    SwitchButton,
    setTheme,
    Theme,
    ListWidget,
    PrimaryToolButton,
)
from model import process_images

CONFIG_FILE = pathlib.Path("labels.json")
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r+") as f:
        CONFIG_DATA = json.load(f)

else:
    data = {"labels": ["anime"]}
    with open(CONFIG_FILE, "a+") as f:
        json.dump(data, f, indent=4)
        f.seek(0)
        CONFIG_DATA = json.load(f)


class Worker(QObject):
    finished = Signal()
    # A signal to report errors back to the main thread
    error = Signal(str)

    def __init__(self, target_function, *args, **kwargs):
        super().__init__()
        self._target_function = target_function
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self._target_function(*self._args, **self._kwargs)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class HLayout(QFrame):
    """Base widget for sub-interfaces"""

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = TitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)

        setFont(self.label, 24)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)

        # Must set a globally unique object name for the sub-interface
        self.setObjectName(text.replace(" ", "-"))


class VLayout(QFrame):
    """Base widget for sub-interfaces"""

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = TitleLabel(text, self)
        self.vBoxLayout = QVBoxLayout(self)
        setFont(self.label, 24)
        self.label.setAlignment(Qt.AlignCenter)
        self.vBoxLayout.addWidget(self.label)
        # Must set a globally unique object name for the sub-interface
        self.setObjectName(text.replace(" ", "-"))


class HomeInterface(VLayout):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)

        # Variable to store the selected folder path
        self.folder_path = "No folder selected"

        # Create main layout container
        self.filePathBox = QVBoxLayout()
        self.runBtn = PrimaryPushButton("Run")
        # Create folder selection widgets
        self.folderLabel = SubtitleLabel(f"Selected Folder: {self.folder_path}", self)
        self.folderLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.folderLabel.setWordWrap(True)
        self.selectFolderButton = PrimaryPushButton("Select Folder")
        self.selectFolderButton.clicked.connect(self._open_folder_dialog)
        self.runBtn.clicked.connect(self._start_thread)
        self.filePathBox.addWidget(self.folderLabel, 0, Qt.AlignmentFlag.AlignCenter)
        self.filePathBox.addWidget(
            self.selectFolderButton, 0, Qt.AlignmentFlag.AlignCenter
        )
        self.filePathBox.addWidget(self.runBtn, 0, Qt.AlignmentFlag.AlignCenter)
        self.vBoxLayout.addLayout(self.filePathBox)
        self.vBoxLayout.addStretch(1)

    def _open_folder_dialog(self):
        """Opens folder selection dialog and updates the label"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            "",
            QFileDialog.ShowDirsOnly,
        )

        if folder:
            self.folder_path = folder
            self.folderLabel.setText(f"Selected Folder: {self.folder_path}")
            print(f"Folder selected: {self.folder_path}")
        else:
            print("Folder selection cancelled")

    def _handle_error(self, message):
        """
        Handles errors reported from the worker thread.
        """
        QMessageBox.critical(self, "Thread Error", f"An error occurred: {message}")

    def _start_thread(self):
        """
        Initiates the function call on a new thread with error checking.
        """
        # 1. Error handling for empty folder path or labels
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "Please select a folder.")
            return
        if not CONFIG_DATA["labels"]:
            QMessageBox.warning(
                self, "Error", "No labels found. Please configure labels."
            )
            return

        # 2. Setup the worker and thread
        self.thread = QThread()
        # Pass the function 'f' and its arguments to the worker
        self.worker = Worker(process_images, self.folder_path, CONFIG_DATA["labels"])

        # 3. Move the worker to the new thread
        self.worker.moveToThread(self.thread)

        # 4. Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self._handle_error)

        # 5. Start the thread
        self.thread.start()


class SettingInterface(VLayout):
    def __init__(self, text: str, parent=None):
        self.is_dark_mode = False
        super().__init__(text, parent)
        self.darkModeBox = QVBoxLayout()
        toggleDarkModeButton = SwitchButton()
        toggleDarkModeButton.checkedChanged.connect(
            lambda checked: self._update_dark_mode(checked)
        )
        self.darkModeLabel = SubtitleLabel("Dark Mode", self)
        self.darkModeBox.addWidget(self.darkModeLabel, 0)
        self.darkModeBox.addWidget(toggleDarkModeButton, 0)
        self.vBoxLayout.addLayout(self.darkModeBox)
        self.vBoxLayout.addStretch(1)
        print(toggleDarkModeButton.isChecked())

    def _update_dark_mode(self, checked):
        """Updates dark mode state and checks if True"""
        self.is_dark_mode = checked
        print("Is the button selected:", checked)
        if checked:
            print("Dark mode is Off")
            setTheme(Theme.DARK)
        else:
            print("Dark mode is ON")
            setTheme(Theme.LIGHT)


class LabelsInterface(VLayout):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._labels = CONFIG_DATA["labels"]
        self.Labels = ListWidget()
        for label in self._labels:
            item = QListWidgetItem(label)
            item.setIcon(QIcon(":/qfluentwidgets/images/logo.png"))
            self.Labels.addItem(item)
        self.labelsBox = QVBoxLayout()
        self.addLabelBtn = PrimaryToolButton(FIF.ADD)
        self.labelsBox.addWidget(self.Labels)
        self.vBoxLayout.addLayout(self.labelsBox)
        self.hBoxLayout = QHBoxLayout()
        self.hBoxLayout.addStretch()
        self.hBoxLayout.addWidget(self.addLabelBtn)
        self.vBoxLayout.addLayout(self.hBoxLayout)


class MainWindow(FluentWindow):
    """Main Interface with Navigation"""

    def __init__(self):
        super().__init__()
        self.homeInterface = HomeInterface("Home", self)
        self.labelsInterface = LabelsInterface("Labels", self)
        self.settingInterface = SettingInterface("Settings", self)
        self.initNavigation()
        self.initWindow()

    def initNavigation(self):
        self.addSubInterface(self.homeInterface, FIF.HOME, "Home")
        self.addSubInterface(self.labelsInterface, FIF.TAG, "Labels")
        self.addSubInterface(
            self.settingInterface,
            FIF.SETTING,
            "Settings",
            NavigationItemPosition.BOTTOM,
        )

    def initWindow(self):
        self.resize(900, 700)
        self.setWindowIcon(QIcon(":/qfluentwidgets/images/logo.png"))
        self.setWindowTitle("Bullseye🎯")


class BullseyeApp:
    """Application wrapper for better structure"""

    def __init__(self, args):
        self.app = QApplication(args)
        self.window = MainWindow()

    def run(self):
        self.window.show()
        self.app.exec()


if __name__ == "__main__":
    app = BullseyeApp(sys.argv)
    app.run()

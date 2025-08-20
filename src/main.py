import json
import os
import pathlib
import sys

import torch
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout,
                               QListWidgetItem, QMessageBox, QVBoxLayout)
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (FluentWindow, ListWidget, NavigationItemPosition,
                            PrimaryPushButton, PrimaryToolButton,
                            SubtitleLabel, SwitchButton, Theme, TitleLabel,
                            setFont, setTheme)

from model import create_ort_session, load_model, process_images

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


class AppState(QObject):
    """Central application state container. Notifies listeners of changes."""

    modelChanged = Signal(object)  
    ortSessionChanged = Signal(object)
    deviceChanged = Signal(object)  

    def __init__(self):
        super().__init__()
        self._model = None
        self._ort_session = None
        self._device = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if self._model != value:
            self._model = value
            self.modelChanged.emit(value) 
    @property
    def ort_session(self):
        return self._ort_session

    @ort_session.setter
    def ort_session(self, value):
        if self._ort_session != value:
            self._ort_session = value
            self.ortSessionChanged.emit(value)  

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if self._device != value:
            self._device = value
            self.deviceChanged.emit(value) 


class Worker(QObject):
    finished = Signal()
    result = Signal(object)
    error = Signal(str)

    def __init__(self, target_function, *args, **kwargs):
        super().__init__()
        self._target_function = target_function
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            r = self._target_function(*self._args, **self._kwargs)
            self.result.emit(r)
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
        self.appState = parent.appState
        # Variable to store the selected folder path
        self.folder_path = "No folder selected"

        # Create main layout container
        self.filePathBox = QVBoxLayout()
        self.loading_icon = SubtitleLabel("Loading model...")
        self.vBoxLayout.addWidget(self.loading_icon)
        self.runBtn = PrimaryPushButton("Run")
        # Create folder selection widgets
        self.folderLabel = SubtitleLabel(f"Selected Folder: {self.folder_path}", self)
        self.folderLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.folderLabel.setWordWrap(False)
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
        self.appState.modelChanged.connect(self.on_model_loaded)

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
        self.worker = Worker(
            process_images,
            self.folder_path,
            CONFIG_DATA["labels"],
            self.appState.device,
            self.appState.model,
            self.appState.ort_session,
        )

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

    def on_model_loaded(self, model):
        # This slot is called automatically when app_state.model is set.
        if model is not None:
            # Model is loaded! Hide loading, show content.
            self.loading_icon.setText("Model Loaded")
            # You can also do other setup here with the model
        else:
            # Model was set to None (e.g., on error), show loading again.
            self.loading_icon.setText("Loading Model")


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
    def __init__(self, text: str, parent):
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

    def __init__(self, appState):
        super().__init__()
        self.appState = appState
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
        self.appState = AppState()
        self.window = MainWindow(self.appState)
        self.appState.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thread = QThread()
        self.worker = Worker(self.load_everything)
        self.worker.result.connect(self._handle_loaded)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.error.connect(self._handle_error)
        self.thread.start()

    def load_everything(self):
        device = self.appState.device
        model, preprocess = load_model(device)
        ort_session = create_ort_session(model, device)
        return {"model": model, "ort_session": ort_session}

    def _handle_error(self, message):
        """
        Handles errors reported from the worker thread.
        """
        print(message)

    def _handle_loaded(self, result):
        """Slot to receive the loaded model from the worker thread."""
        model = result["model"]
        ort_session = result["ort_session"]
        print("Model loaded received in main thread!")
        if model != None:
            self.appState.model = model
        print("ORT Session created!")
        if ort_session != None:
            self.appState.ort_session = ort_session

    def run(self):
        self.window.show()
        self.app.exec()


if __name__ == "__main__":
    app = BullseyeApp(sys.argv)
    app.run()

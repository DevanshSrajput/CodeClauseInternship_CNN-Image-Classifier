import sys
import os
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                             QTabWidget, QFileDialog, QProgressBar, QMessageBox, QGroupBox,
                             QRadioButton, QButtonGroup, QScrollArea, QSizePolicy, QInputDialog,
                             QCheckBox, QSlider, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

from src.model import CIFAR10CNN, FashionMNISTCNN
from src.dataset import get_dataset, get_data_loaders
from src.train import train_model, evaluate_model
from src.utils import (load_model, preprocess_image, predict_image, 
                      plot_training_history, plot_prediction_bar)

class TrainingThread(QThread):
    update_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, model, train_loader, val_loader, device, epochs, lr):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
    
    def callback(self, event_type, data):
        if event_type == 'progress' or event_type == 'epoch':
            self.update_signal.emit(data)
        elif event_type == 'complete':
            self.finished_signal.emit(data)
    
    def run(self):
        _, _ = train_model(
            self.model,
            self.train_loader,
            self.val_loader,
            self.device,
            epochs=self.epochs,
            lr=self.lr,
            callback=self.callback
        )

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

class ImageClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("CNN Image Classifier")
        self.setMinimumSize(1000, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border: 1px solid #cccccc;
                border-bottom-color: white;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0d8bf2;
            }
            QPushButton:pressed {
                background-color: #0277bd;
            }
            QPushButton:disabled {
                background-color: #b3e5fc;
            }
            QLabel {
                font-size: 12px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize variables
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = None
        self.history = None
        self.training_thread = None
        self.current_dataset = None
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Create tabs
        self.training_tab = QWidget()
        self.prediction_tab = QWidget()
        
        tabs.addTab(self.training_tab, "Train Model")
        tabs.addTab(self.prediction_tab, "Predict Images")
        
        # Set up training tab
        self.setup_training_tab()
        
        # Set up prediction tab
        self.setup_prediction_tab()
    
    def setup_training_tab(self):
        layout = QVBoxLayout(self.training_tab)
        
        # Dataset selection
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QHBoxLayout(dataset_group)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["CIFAR-10", "Fashion-MNIST"])
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        
        dataset_layout.addWidget(QLabel("Select Dataset:"))
        dataset_layout.addWidget(self.dataset_combo)
        dataset_layout.addStretch()
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QHBoxLayout(params_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(64)
        
        params_layout.addWidget(QLabel("Epochs:"))
        params_layout.addWidget(self.epochs_spin)
        params_layout.addWidget(QLabel("Learning Rate:"))
        params_layout.addWidget(self.lr_spin)
        params_layout.addWidget(QLabel("Batch Size:"))
        params_layout.addWidget(self.batch_size_spin)
        params_layout.addStretch()
        
        # Training controls
        controls_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.on_train_clicked)
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.stop_button.setEnabled(False)
        
        controls_layout.addWidget(self.train_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()
        
        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready to train...")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        
        # Training visualizations
        viz_layout = QHBoxLayout()
        
        # Create matplotlib canvases for loss and accuracy plots
        self.loss_canvas = MatplotlibCanvas(width=5, height=4)
        self.acc_canvas = MatplotlibCanvas(width=5, height=4)
        
        viz_layout.addWidget(self.loss_canvas)
        viz_layout.addWidget(self.acc_canvas)
        
        # Add components to layout
        layout.addWidget(dataset_group)
        layout.addWidget(params_group)
        layout.addLayout(controls_layout)
        layout.addWidget(progress_group)
        layout.addLayout(viz_layout)
        
        # Initialize plots
        self.init_plots()
    
    def setup_prediction_tab(self):
        layout = QVBoxLayout(self.prediction_tab)
        
        # Model loading
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout(model_group)
        
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.on_load_model_clicked)
        
        self.model_path_label = QLabel("No model loaded")
        
        model_layout.addWidget(self.load_model_button)
        model_layout.addWidget(self.model_path_label)
        model_layout.addStretch()
        
        # Image selection
        image_group = QGroupBox("Image")
        image_layout = QHBoxLayout(image_group)
        
        self.load_image_button = QPushButton("Upload Image")
        self.load_image_button.clicked.connect(self.on_load_image_clicked)
        self.load_image_button.setEnabled(False)
        
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.on_predict_clicked)
        self.predict_button.setEnabled(False)
        
        image_layout.addWidget(self.load_image_button)
        image_layout.addWidget(self.predict_button)
        image_layout.addStretch()
        
        # Preprocessing options
        preproc_group = QGroupBox("Preprocessing Options")
        preproc_layout = QVBoxLayout(preproc_group)
        
        self.enhance_contrast_cb = QCheckBox("Enhance Contrast")
        self.enhance_contrast_cb.setChecked(True)
        
        self.threshold_cb = QCheckBox("Apply Thresholding")
        self.threshold_cb.setChecked(True)
        
        self.invert_cb = QCheckBox("Invert Colors")
        
        preproc_layout.addWidget(self.enhance_contrast_cb)
        preproc_layout.addWidget(self.threshold_cb)
        preproc_layout.addWidget(self.invert_cb)
        
        # Image and prediction display
        display_layout = QHBoxLayout()
        
        # Image display
        image_display_group = QGroupBox("Input Image")
        image_display_layout = QVBoxLayout(image_display_group)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("background-color: white; border: 1px solid #cccccc;")
        
        image_display_layout.addWidget(self.image_label)
        
        # Prediction results
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout(results_group)
        
        self.prediction_label = QLabel("Upload an image and click 'Predict'")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setWordWrap(True)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        
        # Create matplotlib canvas for prediction probabilities
        self.prediction_canvas = MatplotlibCanvas(width=5, height=4)
        
        results_layout.addWidget(self.prediction_label)
        results_layout.addWidget(self.confidence_label)
        results_layout.addWidget(self.prediction_canvas)
        
        # Add components to layout
        display_layout.addWidget(image_display_group)
        display_layout.addWidget(results_group)
        
        layout.addWidget(model_group)
        layout.addWidget(image_group)
        layout.addWidget(preproc_group)
        layout.addLayout(display_layout)
    
    def init_plots(self):
        # Initialize loss plot
        self.loss_canvas.axes.clear()
        self.loss_canvas.axes.set_title('Training and Validation Loss')
        self.loss_canvas.axes.set_xlabel('Epoch')
        self.loss_canvas.axes.set_ylabel('Loss')
        self.loss_canvas.axes.grid(True)
        self.loss_canvas.draw()
        
        # Initialize accuracy plot
        self.acc_canvas.axes.clear()
        self.acc_canvas.axes.set_title('Training and Validation Accuracy')
        self.acc_canvas.axes.set_xlabel('Epoch')
        self.acc_canvas.axes.set_ylabel('Accuracy (%)')
        self.acc_canvas.axes.grid(True)
        self.acc_canvas.draw()
    
    def update_plots(self, history):
        # Update loss plot
        self.loss_canvas.axes.clear()
        self.loss_canvas.axes.plot(history['train_loss'], 'b-', label='Training Loss')
        self.loss_canvas.axes.plot(history['val_loss'], 'r-', label='Validation Loss')
        self.loss_canvas.axes.set_title('Training and Validation Loss')
        self.loss_canvas.axes.set_xlabel('Epoch')
        self.loss_canvas.axes.set_ylabel('Loss')
        self.loss_canvas.axes.legend()
        self.loss_canvas.axes.grid(True)
        self.loss_canvas.draw()
        
        # Update accuracy plot
        self.acc_canvas.axes.clear()
        self.acc_canvas.axes.plot(history['train_acc'], 'b-', label='Training Accuracy')
        self.acc_canvas.axes.plot(history['val_acc'], 'r-', label='Validation Accuracy')
        self.acc_canvas.axes.set_title('Training and Validation Accuracy')
        self.acc_canvas.axes.set_xlabel('Epoch')
        self.acc_canvas.axes.set_ylabel('Accuracy (%)')
        self.acc_canvas.axes.legend()
        self.acc_canvas.axes.grid(True)
        self.acc_canvas.draw()
    
    def on_dataset_changed(self):
        self.current_dataset = self.dataset_combo.currentText()
    
    def on_train_clicked(self):
        # Disable training button during training
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Get training parameters
        dataset_name = self.dataset_combo.currentText()
        epochs = self.epochs_spin.value()
        lr = self.lr_spin.value()
        batch_size = self.batch_size_spin.value()
        
        # Load dataset
        try:
            train_dataset, test_dataset, self.classes = get_dataset(dataset_name)
            self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
                train_dataset, test_dataset, batch_size=batch_size)
            
            # Initialize model
            if dataset_name == "CIFAR-10":
                self.model = CIFAR10CNN().to(self.device)
                self.current_dataset = "CIFAR-10"
            else:
                self.model = FashionMNISTCNN().to(self.device)
                self.current_dataset = "Fashion-MNIST"
            
            # Initialize progress bar
            self.progress_bar.setValue(0)
            total_batches = len(self.train_loader)
            self.progress_bar.setMaximum(total_batches * epochs)
            
            # Initialize training history
            self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            
            # Start training thread
            self.training_thread = TrainingThread(
                self.model, self.train_loader, self.val_loader, self.device, epochs, lr)
            self.training_thread.update_signal.connect(self.on_training_update)
            self.training_thread.finished_signal.connect(self.on_training_finished)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading dataset: {str(e)}")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def on_stop_clicked(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
            self.progress_label.setText("Training stopped by user")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def on_training_update(self, data):
        if 'epoch' in data:
            epoch = data['epoch']
            train_loss = data['train_loss']
            train_acc = data['train_acc']
            val_loss = data.get('val_loss', 0)
            val_acc = data.get('val_acc', 0)
            
            # Update progress label
            self.progress_label.setText(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"Train Acc = {train_acc:.2f}%, Val Loss = {val_loss:.4f}, "
                f"Val Acc = {val_acc:.2f}%"
            )
            
            # Update history
            if len(self.history['train_loss']) < epoch:
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Update plots
                self.update_plots(self.history)
        
        elif 'batch' in data:
            epoch = data['epoch']
            batch = data['batch']
            total_batches = data['total_batches']
            train_loss = data['train_loss']
            train_acc = data['train_acc']
            
            # Update progress bar
            progress = (epoch - 1) * total_batches + batch
            self.progress_bar.setValue(progress)
            
            # Update progress label
            self.progress_label.setText(
                f"Epoch {epoch}, Batch {batch}/{total_batches}: "
                f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%"
            )
            
            # Force the application to process events to update the UI
            QApplication.processEvents()
    
    def on_training_finished(self, data):
        model_path = data['path']
        self.history = data['history']
        
        self.progress_label.setText(f"Training complete! Model saved to: {model_path}")
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        # Update plots with final history
        self.update_plots(self.history)
        
        # Re-enable training button
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Enable prediction
        self.model_path_label.setText(f"Model: {os.path.basename(model_path)}")
        self.load_image_button.setEnabled(True)
        
        # Show completion message
        QMessageBox.information(self, "Training Complete", 
                              f"Model trained successfully!\n"
                              f"Final accuracy: {self.history['val_acc'][-1]:.2f}%\n"
                              f"Model saved to: {model_path}")
    
    def on_load_model_clicked(self):
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(
            self, "Load Model", "./models", "PyTorch Models (*.pth)")
        
        if model_path:
            try:
                # Ask user to select dataset type
                dataset_options = ["CIFAR-10", "Fashion-MNIST"]
                selected_dataset, ok = QInputDialog.getItem(
                    self, "Select Dataset", "Model was trained on:", dataset_options, 0, False)
                
                if not ok:
                    return
                
                if selected_dataset == "CIFAR-10":
                    self.model = load_model(CIFAR10CNN, model_path, self.device)
                    self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
                                  'dog', 'frog', 'horse', 'ship', 'truck')
                else:
                    self.model = load_model(FashionMNISTCNN, model_path, self.device)
                    self.classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
                
                self.current_dataset = selected_dataset
                self.model_path_label.setText(f"Model: {os.path.basename(model_path)}")
                self.load_image_button.setEnabled(True)
                
                QMessageBox.information(self, "Model Loaded", f"Model loaded successfully!")
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading model: {str(e)}")
    
    def on_load_image_clicked(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        
        if image_path:
            try:
                # Display the image
                pixmap = QPixmap(image_path)
                
                # Scale the image to fit the label while preserving aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.image_label.width(), self.image_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                
                self.image_label.setPixmap(scaled_pixmap)
                self.image_path = image_path
                self.predict_button.setEnabled(True)
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
    
    def on_predict_clicked(self):
        if not hasattr(self, 'image_path') or not self.model:
            return
        
        try:
            # Load and show the original image for reference
            orig_img = Image.open(self.image_path)
            
            # Get preprocessing options
            enhance_contrast = self.enhance_contrast_cb.isChecked()
            apply_threshold = self.threshold_cb.isChecked()
            invert_colors = self.invert_cb.isChecked()
            
            # Custom preprocessing for Fashion-MNIST
            if self.current_dataset == "Fashion-MNIST":
                # Convert to grayscale
                img = Image.open(self.image_path).convert('L')
                
                # Apply preprocessing based on user options
                from torchvision import transforms
                
                transform_list = [
                    transforms.Resize((28, 28)),
                    transforms.CenterCrop((28, 28))
                ]
                
                if enhance_contrast:
                    transform_list.append(transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 1.5)))
                    transform_list.append(transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 0.8)))
                
                transform_list.append(transforms.ToTensor())
                transform_list.append(transforms.Normalize((0.2860,), (0.3530,)))
                
                if apply_threshold:
                    transform_list.append(transforms.Lambda(lambda x: (x > 0.1).float()))
                
                if invert_colors:
                    transform_list.append(transforms.Lambda(lambda x: 1.0 - x))
                
                custom_transform = transforms.Compose(transform_list)
                image_tensor = custom_transform(img).unsqueeze(0)
            else:
                # Use standard preprocessing for CIFAR-10
                image_tensor = preprocess_image(self.image_path, self.current_dataset)
            
            # Debug: Show the preprocessed image
            if self.current_dataset == "Fashion-MNIST":
                # Create a figure to display the preprocessed image
                debug_fig = plt.figure(figsize=(3, 3))
                plt.imshow(image_tensor.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
                plt.title("Preprocessed Image")
                plt.axis('off')
                
                # Convert to QPixmap and display
                debug_canvas = FigureCanvas(debug_fig)
                debug_canvas.draw()
                width, height = debug_canvas.get_width_height()
                image = QImage(debug_canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)
                
                # Show in a small popup to help with debugging
                debug_label = QLabel()
                debug_label.setPixmap(QPixmap.fromImage(image))
                debug_window = QDialog(self)
                debug_window.setWindowTitle("Preprocessed Image")
                debug_layout = QVBoxLayout(debug_window)
                debug_layout.addWidget(debug_label)
                debug_window.setLayout(debug_layout)
                debug_window.show()
            
            # Make prediction
            prediction = predict_image(self.model, image_tensor, self.device, self.classes)
            
            # Display results
            class_name = prediction['class_name']
            confidence = prediction['confidence'] * 100
            
            self.prediction_label.setText(f"<h2>Prediction: {class_name}</h2>")
            self.confidence_label.setText(f"<h3>Confidence: {confidence:.2f}%</h3>")
            
            # Create bar chart for class probabilities
            self.prediction_canvas.axes.clear()
            
            # Plot horizontal bars
            probs = prediction['all_probs']
            y_pos = np.arange(len(self.classes))
            bars = self.prediction_canvas.axes.barh(y_pos, probs, align='center')
            self.prediction_canvas.axes.set_yticks(y_pos)
            self.prediction_canvas.axes.set_yticklabels(self.classes)
            self.prediction_canvas.axes.invert_yaxis()  # Labels read top-to-bottom
            self.prediction_canvas.axes.set_xlabel('Probability')
            self.prediction_canvas.axes.set_title('Class Probabilities')
            
            # Highlight predicted class
            bars[prediction['class_idx']].set_color('green')
            
            self.prediction_canvas.draw()
            
            # Print probabilities to console for debugging
            print("Class probabilities:")
            for i, (cls, prob) in enumerate(zip(self.classes, probs)):
                print(f"{i}. {cls}: {prob*100:.2f}%")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error making prediction: {str(e)}")
            import traceback
            traceback.print_exc()
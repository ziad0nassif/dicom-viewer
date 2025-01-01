import sys
import os
import cv2

import random
import string
import pydicom
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QTextEdit, QWidget, 
                             QLineEdit, QMessageBox, QSlider, QTableWidget, 
                             QTableWidgetItem, QDialog, QScrollArea,QInputDialog,QTabWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer

class DicomViewer(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel for file operations
        left_panel = QVBoxLayout()
    
        # Folder Open Button
        open_folder_button = QPushButton("Open DICOM Folder")
        open_folder_button.clicked.connect(self.open_dicom_folder)
        left_panel.addWidget(open_folder_button)
        
        # Single M2D File Open Button
        open_m2d_button = QPushButton("Open M2D DICOM File")
        open_m2d_button.clicked.connect(self.open_m2d_file)
        left_panel.addWidget(open_m2d_button)
        
        # Slider for image navigation
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.display_current_image)
        left_panel.addWidget(self.slider)

        # Playback controls
        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.stop_btn = QPushButton("Stop")
        self.frame_rate_input = QLineEdit()
        self.frame_rate_input.setPlaceholderText("Frame Rate (fps)")
        self.frame_rate_input.setText("5")  # Default frame rate

        self.play_btn.clicked.connect(self.start_playback)
        self.stop_btn.clicked.connect(self.stop_playback)

        # Initially disable play/stop buttons
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.stop_btn)
        playback_layout.addWidget(self.frame_rate_input)
        left_panel.addLayout(playback_layout)
        
        # Tag Exploration Button
        tag_exploration_btn = QPushButton("Explore DICOM Tags")
        tag_exploration_btn.clicked.connect(self.show_dicom_tags)
        left_panel.addWidget(tag_exploration_btn)
        
        # Anonymization section
        anonymize_layout = QHBoxLayout()
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("Anonymization Prefix")
        anonymize_btn = QPushButton("Anonymize DICOM")
        anonymize_btn.clicked.connect(self.anonymize_dicom)
        anonymize_layout.addWidget(self.prefix_input)
        anonymize_layout.addWidget(anonymize_btn)
        left_panel.addLayout(anonymize_layout)

        # New button for creating 3D montage
        self.load_3d_montage_btn = QPushButton("Create 3D Montage")
        self.load_3d_montage_btn.clicked.connect(self.create_current_series_montage)
        self.load_3d_montage_btn.setEnabled(False)  # Disable until DICOM series is loaded
        left_panel.addWidget(self.load_3d_montage_btn)
        
        # Right panel with tabs
        right_panel = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs for single view and montage
        self.single_view_tab = QWidget()
        self.montage_tab = QWidget()
        
        # Setup single view tab
        single_view_layout = QVBoxLayout()
        self.slice_label = QLabel("DICOM image will appear here")
        self.slice_label.setAlignment(Qt.AlignCenter)
        self.slice_label.setFixedSize(800, 800)
        single_view_layout.addWidget(self.slice_label)
        self.single_view_tab.setLayout(single_view_layout)
        
        # Setup montage tab
        montage_layout = QVBoxLayout()
        self.montage_scroll_area = QScrollArea()
        self.montage_label = QLabel("3D Montage will appear here")
        self.montage_label.setAlignment(Qt.AlignCenter)
        self.montage_scroll_area.setWidget(self.montage_label)
        self.montage_scroll_area.setWidgetResizable(True)
        montage_layout.addWidget(self.montage_scroll_area)
        self.montage_tab.setLayout(montage_layout)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.single_view_tab, "Single View")
        self.tab_widget.addTab(self.montage_tab, "Montage View")
        
        # Add tab widget to right panel
        right_panel.addWidget(self.tab_widget)
        
        # Add panels to main layout
        left_container = QWidget()
        left_container.setLayout(left_panel)
        left_container.setFixedWidth(300)
        main_layout.addWidget(left_container)
        
        right_container = QWidget()
        right_container.setLayout(right_panel)
        main_layout.addWidget(right_container)
        
        # DICOM related attributes
        self.dicom_files = []
        self.current_dicom_dataset = None
        self.current_3d_dataset = None
        self.current_image_index = 0
        self.is_m2d = False
        self.m2d_frames = None
        
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_frame)

    def open_m2d_file(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, "Select M2D DICOM File", "", "DICOM files (*.dcm)")[0]
        
        if file_path:
            try:
                ds = pydicom.dcmread(file_path)
                if hasattr(ds, 'NumberOfFrames'):
                    print(f"Number of Frames: {ds.NumberOfFrames}")
                    print(f"Pixel Array Shape: {ds.pixel_array.shape}")
                    print(f"Pixel Array Dtype: {ds.pixel_array.dtype}")
                    
                    self.current_dicom_dataset = ds
                    self.is_m2d = True
                    self.m2d_frames = ds.pixel_array
                    self.slider.setRange(0, int(ds.NumberOfFrames) - 1)
                    self.current_image_index = 0
                    
                    # Reset series-related attributes
                    self.dicom_files = []
                    
                    # Enable playback controls
                    self.play_btn.setEnabled(True)
                    self.stop_btn.setEnabled(True)
                    
                    # Display first frame
                    self.display_current_image()
                else:
                    QMessageBox.warning(self, "Invalid File", "Selected file is not a multi-frame DICOM.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load M2D DICOM file: {str(e)}")



    def start_playback(self):
        try:
            frame_rate = float(self.frame_rate_input.text())
            interval = int(1000 / frame_rate)  # Convert fps to milliseconds
            self.playback_timer.start(interval)
            self.play_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid frame rate.")

    def stop_playback(self):
        self.playback_timer.stop()
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def advance_frame(self):
        
        current_value = self.slider.value()
        
        self.slider.setValue(current_value + 1)
      

        
        self.display_current_image()


 
    def open_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.dicom_files = [
                os.path.join(folder, f) for f in os.listdir(folder) 
                if f.endswith('.dcm')
            ]
            
            if not self.dicom_files:
                QMessageBox.warning(self, "No DICOM Files", "No DICOM files found in the selected folder.")
                return
            
            # Reset M2D-related attributes
            self.is_m2d = False
            self.m2d_frames = None
            
            self.load_dicom_series()
            
            # Enable 3D montage button
            self.load_3d_montage_btn.setEnabled(True)




    def load_dicom_series(self):
        try:
            # Sort DICOM files by slice location if possible
            self.dicom_files.sort(key=lambda x: pydicom.dcmread(x).get('SliceLocation', 0))
            
            # Load first file to get series details
            self.current_dicom_dataset = pydicom.dcmread(self.dicom_files[0])
            
            # Setup slider for navigation
            self.slider.setRange(0, len(self.dicom_files) - 1)
            self.current_image_index = 0
            
            # Enable play/stop buttons
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)

            # Display first slice
            self.display_current_image()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load DICOM series: {str(e)}")

    def display_current_image(self):
        try:
            # Determine the image source based on M2D or series
            if self.is_m2d and self.m2d_frames is not None:
                # Get the current frame from multi-frame DICOM
                image_array = self.m2d_frames[self.slider.value()]
            elif self.dicom_files:
                # Get the current file from DICOM series
                current_file = self.dicom_files[self.slider.value()]
                ds = pydicom.dcmread(current_file)
                image_array = ds.pixel_array
            else:
                # No images loaded
                return

            # Normalize and process the image array
            def normalize_image(img):
                # Flatten multi-dimensional arrays
                while len(img.shape) > 2:
                    # Prioritize keeping the largest 2D slice
                    if img.shape[0] < img.shape[-1]:
                        img = img[0]  # Take first channel
                    else:
                        img = img[:, :, 0]  # Take first slice

                # Ensure 2D array
                if len(img.shape) != 2:
                    raise ValueError(f"Could not reduce image to 2D. Current shape: {img.shape}")

                # Normalize to 8-bit grayscale
                img_min = img.min()
                img_max = img.max()
                
                # Handle different data types
                if np.issubdtype(img.dtype, np.floating):
                    # For floating point images
                    normalized = ((img - img_min) / (img_max - img_min) * 255).clip(0, 255).astype(np.uint8)
                else:
                    # For integer images
                    normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                
                return normalized

            # Normalize the image
            processed_image = normalize_image(image_array)

            # Create QImage
            height, width = processed_image.shape
            bytes_per_line = width
            q_image = QImage(processed_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            # Scale and display
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.slice_label.width(), 
                self.slice_label.height(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )

            # Set the pixmap
            self.slice_label.setPixmap(scaled_pixmap)

        except Exception as e:
            # Detailed error logging
            error_message = f"Image display error: {str(e)}"
            if 'image_array' in locals():
                error_message += f"\nImage array shape: {image_array.shape}"
                error_message += f"\nImage array dtype: {image_array.dtype}"
            
            QMessageBox.critical(self, "Display Error", error_message)
            
    def show_dicom_tags(self):
        if not self.current_dicom_dataset:
            QMessageBox.warning(self, "No DICOM File", "Please load a DICOM file first.")
            return
        
           # Store the original values before swapping
        first_tag = self.current_dicom_dataset.get((0x0010, 0x0010), "").value if (0x0010, 0x0010) in self.current_dicom_dataset else ""
        second_tag = self.current_dicom_dataset.get((0x0008 ,0x1010), "").value if (0x0008, 0x1010) in self.current_dicom_dataset else ""


        # Swap the values in the dataset
        if (0x0010, 0x0010) in self.current_dicom_dataset and (0x0008, 0x1010) in self.current_dicom_dataset:
            self.current_dicom_dataset[0x0010, 0x0010].value = second_tag
            self.current_dicom_dataset[0x0008, 0x1010].value = first_tag
        
        # Create a dialog to display DICOM tags
        tag_dialog = QDialog(self)
        tag_dialog.setWindowTitle("DICOM Tags")
        tag_dialog.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout()
        
        # Search input for specific tag
        search_layout = QHBoxLayout()
        tag_search = QLineEdit()
        tag_search.setPlaceholderText("Search for a specific tag")
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(lambda: self.search_dicom_tag(tag_search.text(), tag_table))
        search_layout.addWidget(tag_search)
        search_layout.addWidget(search_btn)
        layout.addLayout(search_layout)
        
        # Table to display tags
        tag_table = QTableWidget()
        tag_table.setColumnCount(3)
        tag_table.setHorizontalHeaderLabels(["Tag", "Name", "Value"])
        
        # Populate table with all tags
        self.populate_dicom_tags(tag_table)
        
        layout.addWidget(tag_table)
        
        # Buttons for specific group exploration
        group_buttons_layout = QHBoxLayout()
        groups = {
            "Patient": (0x0010, 0x0011),
            "Study": (0x0020, 0x0021),
            "Modality": (0x0008, 0x0009),
            "Physician": (0x0011, 0x0090),
            "Image": (0x0028, 0x0030)
        }
        
        for group_name, (start, end) in groups.items():
            btn = QPushButton(f"Show {group_name} Info")
            btn.clicked.connect(lambda checked, s=start, e=end: self.show_group_tags(s, e, tag_table))
            group_buttons_layout.addWidget(btn)
        
        layout.addLayout(group_buttons_layout)
        
        tag_dialog.setLayout(layout)
        tag_dialog.exec_()

    def switch_patient_physician_names(self, table):
        """
        Switch the values of patient name and physician name tags in the DICOM dataset
        and update the table display
        """
        try:
            # DICOM tags for patient name and physician name
            patient_name_tag = (0x0010, 0x0010)  # Patient's Name
            physician_name_tag = (0x0008, 0x0090)  # Referring Physician's Name
            
            # Get current values
            patient_name = str(self.current_dicom_dataset[patient_name_tag].value)
            physician_name = str(self.current_dicom_dataset[physician_name_tag].value)
            
            # Switch values in the dataset
            self.current_dicom_dataset[patient_name_tag].value = physician_name
            self.current_dicom_dataset[physician_name_tag].value = patient_name
            
            # Update the table display
            self.populate_dicom_tags(table)
            
            QMessageBox.information(self, "Success", "Patient and Physician names switched successfully!")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to switch names: {str(e)}")
            

    def load_3d_dicom_montage(self):
        folder = QFileDialog.getExistingDirectory(self, "Select 3D DICOM Folder")
        if folder:
            # Find DICOM files in the folder
            dicom_files = [
                os.path.join(folder, f) for f in os.listdir(folder) 
                if f.endswith('.dcm')
            ]
            
            if not dicom_files:
                QMessageBox.warning(self, "No DICOM Files", "No DICOM files found in the selected folder.")
                return
            
            try:
                # Read all DICOM files and sort by slice location
                dicom_datasets = []
                for file_path in dicom_files:
                    ds = pydicom.dcmread(file_path)
                    dicom_datasets.append((ds.get('SliceLocation', 0), ds))
                
                # Sort by slice location
                dicom_datasets.sort(key=lambda x: x[0])
                
                # Set current dataset to the first dataset for tag exploration
                if dicom_datasets:
                    self.current_dicom_dataset = dicom_datasets[0][1]
                
                # Extract pixel arrays
                pixel_arrays = [ds[1].pixel_array for ds in dicom_datasets]
                
                # Create 3D montage
                montage = self.create_3d_montage(pixel_arrays)
                
                # Display montage
                self.display_3d_montage(montage)
                
                # Store the complete dataset for potential future use
                self.current_3d_dataset = dicom_datasets
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load 3D DICOM montage: {str(e)}")


    def create_current_series_montage(self):
        if not self.dicom_files:
            QMessageBox.warning(self, "No DICOM Series", "Please load a DICOM series first.")
            return
        
        # Add a dialog to select number of columns
        columns, ok = QInputDialog.getInt(self, "Montage Columns", 
                                        "Enter number of columns:", 
                                        min=1, max=1000, value=8)
        if not ok:
            return
        
        try:
            # Read and sort DICOM files by slice location
            dicom_datasets = []
            for file_path in self.dicom_files:
                ds = pydicom.dcmread(file_path)
                dicom_datasets.append((ds.get('SliceLocation', 0), ds))
            
            # Sort by slice location
            dicom_datasets.sort(key=lambda x: x[0])
            
            # Extract pixel arrays
            pixel_arrays = [ds[1].pixel_array for ds in dicom_datasets]
            
            # Create 3D montage with user-specified columns
            montage = self.create_3d_montage(pixel_arrays, columns=columns)
            
            # Display montage
            self.display_3d_montage(montage, columns=columns)
            
            # Store the dataset for potential future use
            self.current_3d_dataset = dicom_datasets
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create 3D montage: {str(e)}")
                

    def create_3d_montage(self, pixel_arrays, columns):
        """
        Create a grid montage of 3D image slices for vertical scrolling
        
        :param pixel_arrays: List of 2D numpy arrays (image slices)
        :param columns: Number of columns in the grid
        :param max_slice_height: Maximum height for each slice
        :return: Montage image as numpy array
        """
        # Force the scroll area to update its geometry
        self.montage_scroll_area.updateGeometry()
        QApplication.processEvents()
        
        # Calculate dynamic slice dimensions based on number of columns
        screen_width = self.montage_scroll_area.viewport().width()  # Use viewport width instead
        
        # Ensure we have a valid width
        if screen_width <= 0:
            screen_width = self.montage_scroll_area.width() - 40  # Fallback with scrollbar adjustment
        
       
        # For 4+ columns, use the original logic
        max_slice_width = screen_width // columns
        
        # Normalize all images
        normalized_arrays = []
        
        # Normalize and resize images
        for img in pixel_arrays:
            # Flatten multi-dimensional arrays
            while len(img.shape) > 2:
                img = img[:, :, 0]
            
            # Normalize to 8-bit
            img_min = img.min()
            img_max = img.max()
            normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            
            # Resize to consistent width while maintaining aspect ratio
            height, width = normalized.shape
            aspect_ratio = width / height
            
            # For 1-3 columns, ensure the image takes up the full column width
            new_width = max_slice_width
            new_height = int(new_width / aspect_ratio)
            
            # Use OpenCV if available for better resizing, otherwise use numpy
            try:
                import cv2
                resized = cv2.resize(normalized, (new_width, new_height), interpolation=cv2.INTER_AREA)
            except ImportError:
                from scipy.ndimage import zoom
                scale_factor_w = new_width / width
                scale_factor_h = new_height / height
                resized = zoom(normalized, (scale_factor_h, scale_factor_w), order=1)
            
            normalized_arrays.append(resized)
        
        # Calculate grid dimensions
        total_slices = len(normalized_arrays)
        rows = (total_slices + columns - 1) // columns
        
        # Get slice dimensions from the first normalized array
        slice_height = normalized_arrays[0].shape[0]
        slice_width = normalized_arrays[0].shape[1]
        
        # Create blank montage canvas
        montage = np.zeros((rows * slice_height, columns * slice_width), dtype=np.uint8)
        
        # Fill montage
        for i, slice_img in enumerate(normalized_arrays):
            row = i // columns
            col = i % columns
            
            # Ensure the slice fits in its grid cell
            h, w = slice_img.shape
            x_offset = 0  # No horizontal centering for 1-3 columns
            y_offset = (slice_height - h) // 2  # Center vertically
            
            montage[row*slice_height:(row+1)*slice_height, 
                    col*slice_width:(col+1)*slice_width] = 0  # Clear cell
            
            montage[row*slice_height+y_offset:row*slice_height+y_offset+h, 
                    col*slice_width+x_offset:col*slice_width+x_offset+w] = slice_img
        
        return montage

    def display_3d_montage(self, montage, columns=8):
        """
        Display the 3D montage in a vertically scrollable grid layout and switch to montage tab
        
        :param montage: Numpy array representing the montage
        :param columns: Number of columns in the grid
        """
        # Create QImage
        height, width = montage.shape
        bytes_per_line = width
        q_image = QImage(montage.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Create pixmap
        pixmap = QPixmap.fromImage(q_image)
        
        # Update montage label
        self.montage_label.setPixmap(pixmap)
        
        # Ensure the label can be scrolled
        self.montage_label.setScaledContents(False)
        self.montage_label.setFixedSize(pixmap.size())
        
        # Allow vertical scrolling
        self.montage_scroll_area.setWidgetResizable(True)
        self.montage_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.montage_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Switch to montage tab
        self.tab_widget.setCurrentIndex(1)  # Switch to montage tab


    def populate_dicom_tags(self, tag_table):
        if not self.current_dicom_dataset:
            return
        
        tag_table.setRowCount(0)
        for elem in self.current_dicom_dataset:
            if elem.name != '':
                row = tag_table.rowCount()
                tag_table.insertRow(row)
                tag_table.setItem(row, 0, QTableWidgetItem(f"{elem.tag}"))
                tag_table.setItem(row, 1, QTableWidgetItem(elem.name))
                tag_table.setItem(row, 2, QTableWidgetItem(str(elem.value)))

    def search_dicom_tag(self, tag_query, tag_table):
        tag_table.setRowCount(0)
        for elem in self.current_dicom_dataset:
            if tag_query.lower() in elem.name.lower() or tag_query.lower() in str(elem.tag).lower():
                row = tag_table.rowCount()
                tag_table.insertRow(row)
                tag_table.setItem(row, 0, QTableWidgetItem(f"{elem.tag}"))
                tag_table.setItem(row, 1, QTableWidgetItem(elem.name))
                tag_table.setItem(row, 2, QTableWidgetItem(str(elem.value)))

    def show_group_tags(self, start_group, end_group, tag_table):
        tag_table.setRowCount(0)
        for elem in self.current_dicom_dataset:
            if start_group <= elem.tag.group <= end_group:
                row = tag_table.rowCount()
                tag_table.insertRow(row)
                tag_table.setItem(row, 0, QTableWidgetItem(f"{elem.tag}"))
                tag_table.setItem(row, 1, QTableWidgetItem(elem.name))
                tag_table.setItem(row, 2, QTableWidgetItem(str(elem.value)))

    def anonymize_dicom(self):
        if not self.current_dicom_dataset and not self.current_3d_dataset:
            QMessageBox.warning(self, "No DICOM Data", "Please load a DICOM file or series first.")
            return
        
        prefix = self.prefix_input.text().strip()
        if not prefix:
            QMessageBox.warning(self, "Anonymization Error", "Please enter a prefix for anonymization.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Anonymized Files")
        if not output_dir:
            return

        anonymized_dir = os.path.join(output_dir, f"anonymized_{prefix}")
        try:
            if not os.path.exists(anonymized_dir):
                os.makedirs(anonymized_dir)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create output directory: {str(e)}")
            return

        # Tags to anonymize
        anonymization_tags = [
            0x00100010,  # Patient Name
            0x00100020,  # Patient ID
            0x00100030,  # Patient Birth Date
            0x00080090,  # Referring Physician's Name
            0x00081070,  # Operators' Name
        ]

        random_values = {
            tag: f"{prefix}_{(''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))}"
            for tag in anonymization_tags
        }

        try:
            if self.is_m2d:
                # Anonymize single M2D file
                ds = self.current_dicom_dataset
                for tag in anonymization_tags:
                    if tag in ds:
                        ds[tag].value = random_values[tag]
                
                output_path = os.path.join(anonymized_dir, f"anon_m2d.dcm")
                ds.save_as(output_path)
                QMessageBox.information(self, "Anonymization Successful", 
                                    f"Successfully anonymized M2D file.\nSaved to: {output_path}")
            elif self.current_3d_dataset:
                # Anonymize 3D montage series
                for slice_location, ds in self.current_3d_dataset:
                    for tag in anonymization_tags:
                        if tag in ds:
                            ds[tag].value = random_values[tag]
                    
                    output_path = os.path.join(anonymized_dir, f"anon_slice_{slice_location}.dcm")
                    ds.save_as(output_path)
                
                QMessageBox.information(self, "Anonymization Successful",
                                    f"Successfully anonymized 3D montage.\nSaved to: {anonymized_dir}")
            else:
                # Anonymize standard series
                for dicom_file in self.dicom_files:
                    ds = pydicom.dcmread(dicom_file)
                    for tag in anonymization_tags:
                        if tag in ds:
                            ds[tag].value = random_values[tag]
                    
                    output_path = os.path.join(anonymized_dir, f"anon_{os.path.basename(dicom_file)}")
                    ds.save_as(output_path)
                
                QMessageBox.information(self, "Anonymization Successful",
                                    f"Successfully anonymized all files.\nSaved to: {anonymized_dir}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Anonymization failed: {str(e)}")




def main():
    app = QApplication(sys.argv)
    viewer = DicomViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
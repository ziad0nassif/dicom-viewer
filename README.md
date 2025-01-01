# ﻿DICOM Viewer

## Introduction

### A Python-based DICOM Viewer built with PyQt5, enabling users to load, view, and interact with DICOM images, including single-frame and multi-frame (M2D) files. The application also provides tools for metadata exploration, anonymization, and a 3D montage creation.

###### image of program:
<div>
  <img src="https://github.com/user-attachments/assets/6446e1e2-4015-463e-9577-89c37e2a6863" >
</div>

 ## Features

• Can open dicom any file that has 2D, M2D or 3D image(s).

• If M2D: Display the images as tiles if 3D and as video if M2D.

• Allow the user to explore Dicom tags via the following routes:
      Display all the Dicom tags in the file along with their values.
      Search for a specific Dicom tag and display its value.
      Explore the values of the main Dicom Elements of specific groups (Patient, Study, Modality,
      Physician, Image) through button in the main UI.
      
• Allow the user to anonymize the opened file via replacing the critical information
with some random values that starts with a prefix from the user (i.e. The program
takes this prefix from the user before doing the anonymization).

• Slider Navigation: Quickly navigate through DICOM slices or frames using a slider.

• 3D Montage Creation: Generate a montage of the current DICOM series.



## Advanced Tools
• 3D Montage Creation: Generate a montage of the current DICOM series.

• User can choose number of colomns of tiles appear to him before display.


## Requirements
To run this application, make sure the following are installed:

[requirements.txt](https://github.com/ziad0nassif/image-quality-viewer/blob/f24ab222ba4ecc3c1c73b18a9b2f8119fb814c56/requirements.txt) 



## Usage

• Upon running the application, you can use the "Open DICOM Folder "  button  DICOM series and  "Open M2D DICOM File" button for M2D dicom files to choose data from your system.

• The data will be displayed at the selected scale, and you can easily navigate through the Buttons.


## Logging
The program logs user interactions and critical steps, aiding in debugging and problem resolution. Log files are generated to provide insights into the development process.

### Feel free to fork this repository, make improvements, and submit a pull request if you have any enhancements or bug fixes.

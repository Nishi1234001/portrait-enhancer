# Portrait Enhancer: Raw to Studio-Quality Portraits

## Overview

This project enhances raw human portrait images captured under uncontrolled conditions (mobile camera, low light, motion blur, cluttered backgrounds) into studio-quality portraits using computer vision and image processing techniques.

The enhancements include:
- Removal of motion blur
- Background blur (bokeh effect) to simulate studio portrait depth
- Improved face clarity and sharpness
- Contrast enhancement
- Preservation of natural skin texture and facial identity

## Features

- Fast and efficient image enhancement pipeline
- Uses MediaPipe for face segmentation
- OpenCV for image processing (deblurring, background blur)
- Preserves natural appearance while improving overall quality
- Tested on multiple raw images with varied lighting and background conditions

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Nishi1234001/portrait-enhancer.git
   cd portrait-enhancer

2. Create and activate a virtual environment:
    python -m venv venv
    venv\Scripts\activate     # Windows
   
3. Install dependencies:
    pip install -r requirements.txt

Usage
Run the enhancement script on a raw image:
   python scripts/enhance.py --in demo_images/raw1.jpg --out outputs/result1.jpg

Replace input and output paths as needed.

Demo

Input images: Stored in demo_images/
Enhanced output images: Saved in outputs/
See the demo video here: [Demo Video Link](https://drive.google.com/file/d/1l1dFC5kXtOcPaltc38xk4-0U6tcSvVlb/view?usp=sharing)

portrait-enhancer/
│
├── scripts/                # Enhancement script
│   └── enhance.py
├── demo_images/            # Sample raw images
├── outputs/                # Enhanced portraits output
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore

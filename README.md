# COMP3217-CW2-README

### GitHub Repository location:
- https://github.com/ihish52/comp3217-cw2

### Requirements:
- Python 3.6 or later
	- numpy - _pip install numpy_
	- scikit-learn - _pip install sklearn_
	- pandas - _pip install pandas_
	- matplotlib - _pip install matplotlib_
	- PuLP - _pip install pulp_
	- openpyxl for Excel support on Linux - _pip install openpyxl_

# Running main code – detecting manipulated pricing, scheduling, and plotting:
### Windows
- Classify TestingData.txt and output predicted labels to TestingResults.txt:
    - python classify.py
- Compute LP scheduling solution for abnormal data and plot energy usage:
    - python schedule_plot.py
### Linux
- Predict labels, solve LP problem and plot abnormal energy usage (uses same two commands as Windows, but in one bash script):
    - bash classify_schedule_plot.sh


#
### Compare 7 different classification methods (duplicate Table 1 results):
- python compare_classification_methods.py

### Location of plots detected Abnormal by LDA classification:
- comp3217-cw2/plots

### Location of all plots, Normal and Abnormal, using LDA classification:
- comp3217-cw2/all_plots_lda

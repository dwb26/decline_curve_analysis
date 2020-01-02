# Decline curve analysis

This is a mini-project in some data analysis and curve fitting to some anonymous oil well output time series data. The script produces several plots. The first plot visualises the aggregate output quantities for some input wells on a stacked bar chart. The second visualises the monthly aggregate output quanitites and a daily time series output for the input wells. The third then fits curves to the historical data using an Arp's decline curve and a least squares minimisation routine. The fourth plot uses the forecast data to predict the ultimate recoverable volume for each input well and plots this against the aggregate output on the final day of production.  

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
To run the script, download the .xlsx and and .py files and change to this directory in the terminal/command line. To run the script in iPython, enter either "ipython decline_curve.py" or run in iPython using the "%run" magic command. To tweak the current input well options see the bottom 
of decline_curve.py. Wellnames vary between "P001" to "P754" for the historical data and "P001" to "P020" for the forecast data.

### Prerequisites

What things you need to install the software and how to install them
To run the script the latest installs of Python3, numpy, pandas, seaborn, scipy and matplotlib are recommended. These can be downloaded using pip and then, for example,

```
pip install numpy
```

## Authors

* **Dan Burrows** - *Decline curve analysis* - (https://github.com/dwb26)

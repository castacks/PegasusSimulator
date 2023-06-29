import sys
import select
import json
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

# Create a PyQtGraph window for the plot
app = QApplication([])
win = pg.GraphicsLayoutWidget(title="Real-Time Plots")
win.show()  # show the window

# Create a dictionary to store data for each label
data = {}

def read_input():
    # Use select to check if there's data available on stdin
    while select.select([sys.stdin,],[],[],0.0)[0]:
        line = sys.stdin.readline()
        if line:
            data_stream = json.loads(line.strip())

            # If this is the first time we've seen this label, create a plot for it
            if data_stream['label'] not in data:
                # Adjusting the plot title to be bigger and bold using HTML styling
                title_html = f'<span style="font-size: 22pt; font-weight: bold;">{data_stream["label"]}</span>'
                plot = win.addPlot(title=title_html)
                
                # Setting the tick label font size for both axes
                font = QtGui.QFont()
                font.setPixelSize(22)
                plot.getAxis('bottom').setStyle(tickFont=font)
                plot.getAxis('left').setStyle(tickFont=font)
                
                curve = plot.plot(pen='y')

                data[data_stream['label']] = {'values': [], 'plot': plot, 'curve': curve}

                # Stack plots vertically
                if len(data) > 1:
                    win.nextRow()
                    win.addItem(data[data_stream['label']]['plot'])

            # Add the new value to the data for this label
            data[data_stream['label']]['values'].append(data_stream['value'])

# Use a QTimer to call the read_input function periodically
input_timer = QtCore.QTimer()
input_timer.timeout.connect(read_input)
input_timer.start(50)  # Check for new input every 50 ms

def update():
    # Update the plot for each label
    for label in data:
        data[label]['curve'].setData(data[label]['values'])

# Set the update function to be called on a timer every 50ms
plot_timer = QtCore.QTimer()
plot_timer.timeout.connect(update)
plot_timer.start(50)

# Start the PyQtGraph event loop
app.exec_()

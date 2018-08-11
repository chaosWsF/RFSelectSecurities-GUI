import sys
import os
import matplotlib
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from canvasUI import MplCanvas
from mainRF import rf_train
from baseCal import base_a

matplotlib.use('Qt5Agg')
main_dir = '../../'
close_path = '../../##.*'
# TODO change into a date input box
start_date_list = ["201501"]
end_date_list = ["201712"]

try:    # load close prices
    df = pd.read_csv(close_path, delimiter="\t", skiprows=1, index_col=0, parse_dates=True,
                     date_parser=lambda dates: pd.datetime.strptime(dates, '%Y%m%d'))
    standard = df[['000000.AA']]
except FileNotFoundError:
    print('##.* is not found.')


class ApplicationWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        QtWidgets.QMainWindow.__init__(self)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Interface for Random Forest")
        self.main_widget = QtWidgets.QWidget(self)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(2)

        l = QtWidgets.QHBoxLayout(self.main_widget)
        l.addLayout(grid)
        l.setContentsMargins(10, 10, 10, 10)

        self.canvas = MplCanvas(self.main_widget)
        l.addWidget(self.canvas)

        self.combo_for_fea = QtWidgets.QComboBox(self.main_widget)
        self.combo_for_fea.addItem("...")
        self.combo_for_fea.addItem("......")
        grid.addWidget(self.combo_for_fea, 1, 1)

        self.combo_for_period = QtWidgets.QComboBox(self.main_widget)
        self.combo_for_period.addItem("...")
        self.combo_for_period.addItem("......")
        grid.addWidget(self.combo_for_period, 3, 1)

        self.combo_for_method = QtWidgets.QComboBox(self.main_widget)
        self.combo_for_method.addItem("Scikit-Learn")
        self.combo_for_method.addItem("LightGBM")
        grid.addWidget(self.combo_for_method, 5, 1)

        self.stn = QtWidgets.QPushButton('Start')
        grid.addWidget(self.stn, 3, 4)
        self.stn.clicked.connect(self.main_action)

        self.combo_start_date = QtWidgets.QComboBox(self.main_widget)
        self.combo_start_date.addItems(start_date_list)
        grid.addWidget(self.combo_start_date, 7, 1)

        self.combo_end_date = QtWidgets.QComboBox(self.main_widget)
        self.combo_end_date.addItems(end_date_list)
        grid.addWidget(self.combo_end_date, 9, 1)
        
        s_explbl = 'This month is for forecast and will not be calculated when the test is returned.'
        self.explbl = QtWidgets.QLabel('Note: ' + s_explbl)
        grid.addWidget(self.explbl, 10, 1)

        self.p0lb = QtWidgets.QLabel('num_trees')
        grid.addWidget(self.p0lb, 11, 1)
        self.for_num_trees = QtWidgets.QLineEdit('200')
        grid.addWidget(self.for_num_trees, 11, 2, 1, 1)

        self.p1lb = QtWidgets.QLabel('max_depth')
        grid.addWidget(self.p1lb, 13, 1)
        self.for_max_depth = QtWidgets.QLineEdit('7')
        grid.addWidget(self.for_max_depth, 13, 2, 1, 1)

        self.p2lb = QtWidgets.QLabel('min_samples_leaf')
        grid.addWidget(self.p2lb, 15, 1)
        self.for_min_samples_leaf = QtWidgets.QLineEdit('20')
        grid.addWidget(self.for_min_samples_leaf, 15, 2, 1, 1)

        self.lb = QtWidgets.QLabel('num_stocks')
        grid.addWidget(self.lb, 17, 1)
        self.ptxt = QtWidgets.QLineEdit('32')
        grid.addWidget(self.ptxt, 17, 2, 1, 1)

        self.lbl = QtWidgets.QLabel('To be Continued')
        grid.addWidget(self.lbl, 18, 1)
        self.lbl.setFixedSize(200, 300)

        self.qle = QtWidgets.QLineEdit('StockPool\'s Address')
        grid.addWidget(self.qle, 19, 1)
        self.sfn = QtWidgets.QPushButton('Save Pic')
        grid.addWidget(self.sfn, 19, 3, 1, 2)
        self.sfn.clicked.connect(self.save_figure)

    def main_action(self):
        feature_directory = os.path.join(main_dir, 'end', str(self.combo_for_fea.currentText()))
        start_time = str(self.combo_start_date.currentText())
        end_time = str(self.combo_end_date.currentText())
        rfMethod = str(self.combo_for_method.currentText())
        period = str(self.combo_for_period.currentText())
        stock_numbers_sorted = int(str(self.ptxt.text()))
        num_trees = int(str(self.for_num_trees.text()))
        max_d = int(str(self.for_max_depth.text()))
        min_s = int(str(self.for_min_samples_leaf.text()))

        if int(stock_numbers_sorted) != stock_numbers_sorted:
            stock_numbers_sorted = 30

        standard_return, Y_standard, X_standard = base_a(standard, start_date=start_time, end_date=end_time)

        result, x, y, save_path = rf_train(feature_directory, close_price_data=df, data_index=period, 
                                                                    num_trees=num_trees, start_date=start_time, end_date=end_time, 
                                                                    stock_num=stock_numbers_sorted, max_depth=max_d, 
                                                                    min_samples_leaf=min_s, method=rfMethod)

        self.lbl.setText('RF\nAnnual rate of Return: ' + str(round(result, 4)) + 
                                        '%\n\n' + 'Base\nAnnual rate of Return: ' + str(round(standard_return, 4)) + '%')

        self.qle.setText(save_path)
        self.canvas.axes.clear()
        self.canvas.axes.set_xlabel('Date', x=1, y=0)
        self.canvas.axes.set_ylabel('Return', x=0, y=0.95)
        self.canvas.axes.minorticks_on()
        self.canvas.axes.grid(b=True, which='both', linestyle='--')

        self.canvas.axes.plot(x, y, color="red", label="RF", lw=1)
        self.canvas.axes.plot(X_standard, Y_standard, color="blue", label="Base", lw=1)
        self.canvas.axes.axhline(y=1, color='black', lw=1.5)
        self.canvas.axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, 
                                                    shadow=True, ncol=2)
        self.canvas.draw()

    def save_figure(self):
        fig_name = QtWidgets.QFileDialog.getSaveFileName(self, "Saving", filter="*.png")
        if fig_name[0] != '':
            self.canvas.figure.savefig(fig_name[0])
        else:
            pass

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Message', "Do you want to leave?", 
                                                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                                               QtWidgets.QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())

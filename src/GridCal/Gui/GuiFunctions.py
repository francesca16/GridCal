# This file is part of GridCal.
#
# GridCal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GridCal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GridCal.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtGui import *
from warnings import warn

from GridCal.Engine.Devices import BranchTypeConverter, DeviceType, BranchTemplate, BranchType
from GridCal.Engine.io_structures import ResultTypes
from GridCal.Engine.Devices import Bus


class TreeDelegate(QItemDelegate):
    commitData = QtCore.pyqtSignal(object)
    """
    A delegate that places a fully functioning QComboBox in every
    cell of the column to which it's applied
    """
    def __init__(self, parent, data=dict()):
        """
        Constructoe
        :param parent: QTableView parent object
        :param objects: List of objects to set. i.e. [True, False]
        :param object_names: List of Object names to display. i.e. ['True', 'False']
        """
        QItemDelegate.__init__(self, parent)

        # dictionary of lists
        self.data = data

    @QtCore.pyqtSlot()
    def double_click(self):
        print('double clicked!')
        self.commitData.emit(self.sender())

    def createEditor(self, parent, option, index):
        tree = QTreeView(parent)

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['Template'])

        for key in self.data.keys():
            # add parent node
            parent1 = QStandardItem(str(key))

            # add children to parent
            for elm in self.data[key]:
                child1 = QStandardItem(str(elm))
                parent1.appendRow([child1])

            model.appendRow(parent1)

        tree.setModel(model)
        tree.doubleClicked.connect(self.double_click)
        return tree

    def setEditorData(self, editor, index):

        print(editor)
        print(index)

    def setModelData(self, editor, model, index):

        print(editor)
        print(model)
        print(index)

        # model.setData(index, self.object_names[editor.currentIndex()])


class ComboDelegate(QItemDelegate):
    commitData = QtCore.pyqtSignal(object)
    """
    A delegate that places a fully functioning QComboBox in every
    cell of the column to which it's applied
    """
    def __init__(self, parent, objects, object_names):
        """
        Constructoe
        :param parent: QTableView parent object
        :param objects: List of objects to set. i.e. [True, False]
        :param object_names: List of Object names to display. i.e. ['True', 'False']
        """
        QItemDelegate.__init__(self, parent)

        # objects to sent to the model associated to the combobox. i.e. [True, False]
        self.objects = objects

        # object description to display in the combobox. i.e. ['True', 'False']
        self.object_names = object_names

    @QtCore.pyqtSlot()
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.object_names)
        combo.currentIndexChanged.connect(self.currentIndexChanged)
        return combo

    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        val = index.model().data(index)
        idx = self.object_names.index(val)
        editor.setCurrentIndex(idx)
        editor.blockSignals(False)

    def setModelData(self, editor, model, index):
        model.setData(index, self.objects[editor.currentIndex()])


class TextDelegate(QItemDelegate):
    commitData = QtCore.pyqtSignal(object)
    """
    A delegate that places a fully functioning QLineEdit in every
    cell of the column to which it's applied
    """
    def __init__(self, parent):
        """
        Constructoe
        :param parent: QTableView parent object
        """
        QItemDelegate.__init__(self, parent)

    @QtCore.pyqtSlot()
    def returnPressed(self):
        self.commitData.emit(self.sender())

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.returnPressed.connect(self.returnPressed)
        return editor

    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        val = index.model().data(index)
        editor.setText(val)
        editor.blockSignals(False)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.text())


class FloatDelegate(QItemDelegate):
    commitData = QtCore.pyqtSignal(object)
    """
    A delegate that places a fully functioning QDoubleSpinBox in every
    cell of the column to which it's applied
    """
    def __init__(self, parent, min_=-9999, max_=9999):
        """
        Constructoe
        :param parent: QTableView parent object
        """
        QItemDelegate.__init__(self, parent)
        self.min = min_
        self.max = max_

    @QtCore.pyqtSlot()
    def returnPressed(self):
        self.commitData.emit(self.sender())

    def createEditor(self, parent, option, index):
        editor = QDoubleSpinBox(parent)
        editor.setMaximum(self.max)
        editor.setMinimum(self.min)
        editor.setDecimals(8)
        editor.editingFinished.connect(self.returnPressed)
        return editor

    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        val = float(index.model().data(index))
        editor.setValue(val)
        editor.blockSignals(False)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.value())


class ComplexDelegate(QItemDelegate):
    commitData = QtCore.pyqtSignal(object)
    """
    A delegate that places a fully functioning Complex Editor in every
    cell of the column to which it's applied
    """
    def __init__(self, parent):
        """
        Constructoe
        :param parent: QTableView parent object
        """
        QItemDelegate.__init__(self, parent)

    @QtCore.pyqtSlot()
    def returnPressed(self):
        """

        :return:
        """
        self.commitData.emit(self.sender())

    def createEditor(self, parent, option, index):
        """

        :param parent:
        :param option:
        :param index:
        :return:
        """
        editor = QFrame(parent)
        main_layout = QHBoxLayout(editor)
        main_layout.layout().setContentsMargins(0, 0, 0, 0)

        real = QDoubleSpinBox()
        real.setMaximum(9999)
        real.setMinimum(-9999)
        real.setDecimals(8)

        imag = QDoubleSpinBox()
        imag.setMaximum(9999)
        imag.setMinimum(-9999)
        imag.setDecimals(8)

        # button = QPushButton()

        main_layout.addWidget(real)
        main_layout.addWidget(imag)
        # main_layout.addWidget(button)

        # button.clicked.connect(self.returnPressed)

        return editor

    def setEditorData(self, editor, index):
        """

        :param editor:
        :param index:
        :return:
        """
        editor.blockSignals(True)
        val = complex(index.model().data(index))
        editor.children()[1].setValue(val.real)
        editor.children()[2].setValue(val.imag)
        editor.blockSignals(False)

    def setModelData(self, editor, model, index):
        """

        :param editor:
        :param model:
        :param index:
        :return:
        """
        val = complex(editor.children()[1].value(), editor.children()[2].value())
        model.setData(index, val)


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a Qt table view with a pandas data frame
    """
    def __init__(self, data, parent=None, editable=False, editable_min_idx=-1, decimals=6):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.data = np.array(data.values)
        self._cols = data.columns
        self.index = data.index.values
        self.editable = editable
        self.editable_min_idx = editable_min_idx
        self.r, self.c = np.shape(self.data)
        self.isDate = False
        if self.r > 0 and self.c > 0:
            if isinstance(self.index[0], np.datetime64):
                self.index = pd.to_datetime(self.index)
                self.isDate = True

        self.format_string = '.' + str(decimals) + 'f'

        self.formatter = lambda x: "%.2f" % x

    def flags(self, index):
        if self.editable and index.column() > self.editable_min_idx:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.ItemIsEnabled

    def rowCount(self, parent=None):
        return self.r

    def columnCount(self, parent=None):
        return self.c

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                val = self.data[index.row(), index.column()]
                if isinstance(val, str):
                    return val
                if isinstance(val, complex):
                    if val.real != 0 or val.imag != 0:
                        return val.__format__(self.format_string)
                    else:
                        return '0'
                else:
                    if val != 0:
                        return val.__format__(self.format_string)
                    else:
                        return '0'
        return None

    def setData(self, index, value, role=QtCore.Qt.DisplayRole):
        self.data[index.row(), index.column()] = value
        # print("setData", index.row(), index.column(), value)

    def headerData(self, p_int, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._cols[p_int]
            elif orientation == QtCore.Qt.Vertical:
                if self.index is None:
                    return p_int
                else:
                    if self.isDate:
                        return self.index[p_int].strftime('%Y/%m/%d  %H:%M.%S')
                    else:
                        return str(self.index[p_int])
        return None

    def copy_to_column(self, row, col):
        """
        Copies one value to all the column
        @param row: Row of the value
        @param col: Column of the value
        @return: Nothing
        """
        self.data[:, col] = self.data[row, col]

    def get_data(self, mode=None):
        """

        Args:
            mode: 'real', 'imag', 'abs'

        Returns: index, columns, data

        """
        n = len(self._cols)

        if n > 0:
            # gather values
            if type(self._cols) == pd.Index:
                names = self._cols.values

                if len(names) > 0:
                    if type(names[0]) == ResultTypes:
                        names = [val.name for val in names]

            elif type(self._cols) == ResultTypes:
                names = [val.name for val in self._cols]
            else:
                names = [val.name for val in self._cols]

            if self.data.dtype == complex:

                if mode == 'real':
                    values = self.data.real
                elif mode == 'imag':
                    values = self.data.imag
                elif mode == 'abs':
                    values = np.abs(self.data)
                else:
                    values = np.abs(self.data)

            else:
                values = self.data

            return self.index, names, values
        else:
            # there are no elements
            return list(), list(), list()

    def save_to_excel(self, file_name, mode):
        """

        Args:
            file_name:
            mode: 'real', 'imag', 'abs'

        Returns:

        """
        index, columns, data = self.get_data(mode=mode)

        df = pd.DataFrame(data=data, index=index, columns=columns)
        df.to_excel(file_name)

    def copy_to_clipboard(self, mode=None):
        """
        Copy profiles to clipboard
        Args:
            mode: 'real', 'imag', 'abs'
        """
        n = len(self._cols)

        if n > 0:

            index, columns, data = self.get_data(mode=mode)

            data = data.astype(str)

            # header first
            txt = '\t' + '\t'.join(columns) + '\n'

            # data
            for t, index_value in enumerate(index):
                txt += str(index_value) + '\t' + '\t'.join(data[t, :]) + '\n'

            # copy to clipboard
            cb = QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(txt, mode=cb.Clipboard)

        else:
            # there are no elements
            pass


class ObjectsModel(QtCore.QAbstractTableModel):
    """
    Class to populate a Qt table view with the properties of objects
    """
    def __init__(self, objects, editable_headers, parent=None, editable=False,
                 non_editable_attributes=list(), transposed=False, check_unique=list()):
        """

        :param objects: list of objects associated to the editor
        :param editable_headers: Dictionary with the properties and theit units and type {attribute: ('unit', type)}
        :param parent: Parent object: the QTableView object
        :param editable: Is the table editable?
        :param non_editable_attributes: List of attributes that are not enabled for editing
        :param transposed: Display the table transposed?
        """
        QtCore.QAbstractTableModel.__init__(self, parent)

        self.parent = parent

        self.attributes = list(editable_headers.keys())

        self.attribute_types = [editable_headers[attr].tpe for attr in self.attributes]

        self.units = [editable_headers[attr].units for attr in self.attributes]

        self.tips = [editable_headers[attr].definition for attr in self.attributes]

        self.objects = objects

        self.editable = editable

        self.non_editable_attributes = non_editable_attributes

        self.check_unique = check_unique

        self.r = len(self.objects)

        self.c = len(self.attributes)

        self.formatter = lambda x: "%.2f" % x

        self.transposed = transposed

        self.set_delegates()

    def set_delegates(self):
        """
        Set the cell editor types depending on the attribute_types array
        :return:
        """

        if self.transposed:
            F = self.parent.setItemDelegateForRow
        else:
            F = self.parent.setItemDelegateForColumn

        for i in range(self.c):
            tpe = self.attribute_types[i]

            if tpe is bool:
                delegate = ComboDelegate(self.parent, [True, False], ['True', 'False'])
                F(i, delegate)

            if tpe is BranchType:
                conv = BranchTypeConverter(None)
                delegate = ComboDelegate(self.parent, conv.values, conv.options)
                F(i, delegate)

            if tpe is BranchTemplate:
                delegate = ComboDelegate(self.parent, conv.values, conv.options)
                F(i, delegate)

            elif tpe is float:
                delegate = FloatDelegate(self.parent)
                F(i, delegate)

            elif tpe is str:
                delegate = TextDelegate(self.parent)
                F(i, delegate)

            elif tpe is complex:
                delegate = ComplexDelegate(self.parent)
                F(i, delegate)

            elif tpe is None:
                F(i, None)
                if len(self.non_editable_attributes) == 0:
                    self.non_editable_attributes.append(self.attributes[i])
            else:
                pass

    def update(self):
        """
        Add wire
        :param wire:
        :return:
        """
        row = self.rowCount()
        self.beginInsertRows(QtCore.QModelIndex(), row, row)
        # whatever code
        self.endInsertRows()

    def flags(self, index):
        """
        Get the display mode
        :param index:
        :return:
        """
        if self.transposed:
            attr_idx = index.row()
        else:
            attr_idx = index.column()

        if self.editable and self.attributes[attr_idx] not in self.non_editable_attributes:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.ItemIsEnabled

    def rowCount(self, parent=None):
        """
        Get number of rows
        :param parent:
        :return:
        """
        if self.transposed:
            return self.c
        else:
            return self.r

    def columnCount(self, parent=None):
        """
        Get number of columns
        :param parent:
        :return:
        """
        if self.transposed:
            return self.r
        else:
            return self.c

    def data_with_type(self, index):
        """
        Get the data to display
        :param index:
        :return:
        """

        if self.transposed:
            obj_idx = index.column()
            attr_idx = index.row()
        else:
            obj_idx = index.row()
            attr_idx = index.column()

        attr = self.attributes[attr_idx]
        tpe = self.attribute_types[attr_idx]

        if tpe is Bus:
            return getattr(self.objects[obj_idx], attr).name
        elif tpe is BranchType:
            conv = BranchTypeConverter(None)
            return conv.inv_conv[getattr(self.objects[obj_idx], attr)]
        else:
            return getattr(self.objects[obj_idx], attr)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        """
        Get the data to display
        :param index:
        :param role:
        :return:
        """
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self.data_with_type(index))

        return None

    def setData(self, index, value, role=QtCore.Qt.DisplayRole):
        """
        Set data by simple editor (whatever text)
        :param index:
        :param value:
        :param role:
        :return:
        """

        if self.transposed:
            obj_idx = index.column()
            attr_idx = index.row()
        else:
            obj_idx = index.row()
            attr_idx = index.column()

        tpe = self.attribute_types[attr_idx]

        # check taken values
        if self.attributes[attr_idx] in self.check_unique:
            taken = self.attr_taken(self.attributes[attr_idx], value)
        else:
            taken = False

        if not taken:
            if self.attributes[attr_idx] not in self.non_editable_attributes:
                if tpe is BranchType:
                    conv = BranchTypeConverter(None)

                    setattr(self.objects[obj_idx], self.attributes[attr_idx], conv.conv[value.name.lower()])
                else:

                    setattr(self.objects[obj_idx], self.attributes[attr_idx], value)
            else:
                pass  # the column cannot be edited

    def attr_taken(self, attr, val):
        """
        Checks if the attribute value is taken
        :param attr:
        :param val:
        :return:
        """
        for obj in self.objects:
            if val == getattr(obj, attr):
                return True
        return False

    def headerData(self, p_int, orientation, role):
        """
        Get the headers to display
        :param p_int:
        :param orientation:
        :param role:
        :return:
        """
        if role == QtCore.Qt.DisplayRole:

            if self.transposed:
                # for the properties in the schematic view
                if orientation == QtCore.Qt.Horizontal:
                    return 'Value'
                elif orientation == QtCore.Qt.Vertical:
                    if self.units[p_int] != '':
                        return self.attributes[p_int] + ' [' + self.units[p_int] + ']'
                    else:
                        return self.attributes[p_int]
            else:
                # Normal
                if orientation == QtCore.Qt.Horizontal:
                    if self.units[p_int] != '':
                        return self.attributes[p_int] + ' [' + self.units[p_int] + ']'
                    else:
                        return self.attributes[p_int]
                elif orientation == QtCore.Qt.Vertical:
                    return str(p_int) + ':' + str(self.objects[p_int])

        # add a tooltip
        if role == QtCore.Qt.ToolTipRole:
            if p_int < self.c:
                if self.units[p_int] != "":
                    unit = '\nUnits: ' + self.units[p_int]
                else:
                    unit = ''
                return self.attributes[p_int] + unit + ' \n' + self.tips[p_int]
            else:
                # somehow the index is out of range
                return ""

        # if role == QtCore.Qt.DecorationRole:
        #     pixmap = QPixmap(26, 26)
        #     # pixmap.fill(value)
        #     icon = QIcon(pixmap)
        #     return icon

        return None

    def copy_to_column(self, index):
        """
        Copy the value pointed by the index to all the other cells in the column
        :param index: QModelIndex instance
        :return:
        """
        value = self.data_with_type(index=index)
        col = index.column()

        for row in range(self.rowCount()):

            if self.transposed:
                obj_idx = col
                attr_idx = row
            else:
                obj_idx = row
                attr_idx = col

            if self.attributes[attr_idx] not in self.non_editable_attributes:
                setattr(self.objects[obj_idx], self.attributes[attr_idx], value)
            else:
                pass  # the column cannot be edited


class BranchObjectModel(ObjectsModel):

    def __init__(self, objects, editable_headers, parent=None, editable=False,
                 non_editable_attributes=list(), transposed=False, check_unique=list(), catalogue_dict=dict()):

        # type templates catalogue
        self.catalogue_dict = catalogue_dict

        # ObjectsModel.__init__(self, objects, attributes, attr_units, attr_types, parent=parent,
        #                       editable=editable, non_editable_indices=non_editable_indices,
        #                       transposed=transposed, check_unique=check_unique)

        super(BranchObjectModel, self).__init__(objects, editable_headers=editable_headers, parent=parent,
                                                editable=editable, non_editable_attributes=non_editable_attributes,
                                                transposed=transposed, check_unique=check_unique)

    def set_delegates(self):
        """
        Set the cell editor types depending on the attribute_types array
        :return:
        """

        if self.transposed:
            F = self.parent.setItemDelegateForRow
        else:
            F = self.parent.setItemDelegateForColumn

        for i in range(self.c):
            tpe = self.attribute_types[i]

            if tpe is bool:
                delegate = ComboDelegate(self.parent, [True, False], ['True', 'False'])
                F(i, delegate)

            if tpe is BranchType:
                conv = BranchTypeConverter(None)
                delegate = ComboDelegate(self.parent, conv.values, conv.options)
                F(i, delegate)

            elif tpe is BranchTemplate:
                # delegate = TreeDelegate(parent=self.parent, data=self.catalogue_dict)
                delegate = TextDelegate(self.parent)
                F(i, delegate)

            elif tpe is float:
                delegate = FloatDelegate(self.parent)
                F(i, delegate)

            elif tpe is str:
                delegate = TextDelegate(self.parent)
                F(i, delegate)

            elif tpe is complex:
                delegate = ComplexDelegate(self.parent)
                F(i, delegate)

            elif tpe is None:
                F(i, None)
                if len(self.non_editable_attributes) == 0:
                    self.non_editable_attributes.append(self.attributes[i])
            else:
                pass


class ProfilesModel(QtCore.QAbstractTableModel):
    """
    Class to populate a Qt table view with profiles from objects
    """
    def __init__(self, multi_circuit, device_type: DeviceType, magnitude, format, parent):
        """

        Args:
            multi_circuit: MultiCircuit instance
            device_type: string with Load, StaticGenerator, etc...
            magnitude: magnitude to display 'S', 'P', etc...
            parent: Parent object: the QTableView object
        """
        QtCore.QAbstractTableModel.__init__(self, parent)

        self.parent = parent

        self.format = format

        self.circuit = multi_circuit

        self.device_type = device_type

        self.magnitude = magnitude

        self.non_editable_indices = list()

        self.editable = True

        self.r = len(self.circuit.time_profile)

        self.elements = self.circuit.get_elements_by_type(device_type)

        self.c = len(self.elements)

        self.formatter = lambda x: "%.2f" % x

        self.set_delegates()

    def set_delegates(self):
        """
        Set the cell editor types depending on the attribute_types array
        :return:
        """

        if self.format is bool:
            delegate = ComboDelegate(self.parent, [True, False], ['True', 'False'])
            self.parent.setItemDelegate(delegate)

        elif self.format is float:
            delegate = FloatDelegate(self.parent)
            self.parent.setItemDelegate(delegate)

        elif self.format is str:
            delegate = TextDelegate(self.parent)
            self.parent.setItemDelegate(delegate)

        elif self.format is complex:
            delegate = ComplexDelegate(self.parent)
            self.parent.setItemDelegate(delegate)

    def flags(self, index):
        """
        Get the display mode
        :param index:
        :return:
        """

        if self.editable and index.column() not in self.non_editable_indices:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.ItemIsEnabled

    def rowCount(self, parent=None):
        """
        Get number of rows
        :param parent:
        :return:
        """
        return self.r

    def columnCount(self, parent=None):
        """
        Get number of columns
        :param parent:
        :return:
        """
        return self.c

    def data(self, index, role=QtCore.Qt.DisplayRole):
        """
        Get the data to display
        :param index:
        :param role:
        :return:
        """
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                profile_property = self.elements[index.column()].properties_with_profile[self.magnitude]
                array = getattr(self.elements[index.column()], profile_property)
                return str(array[index.row()])

        return None

    def setData(self, index, value, role=QtCore.Qt.DisplayRole):
        """
        Set data by simple editor (whatever text)
        :param index:
        :param value:
        :param role:
        :return:
        """

        if index.column() not in self.non_editable_indices:
            profile_property = self.elements[index.column()].properties_with_profile[self.magnitude]
            getattr(self.elements[index.column()], profile_property)[index.row()] = value
        else:
            pass  # the column cannot be edited

        return True

    def headerData(self, p_int, orientation, role):
        """
        Get the headers to display
        :param p_int:
        :param orientation:
        :param role:
        :return:
        """
        if role == QtCore.Qt.DisplayRole:

            if role == QtCore.Qt.DisplayRole:
                if orientation == QtCore.Qt.Horizontal:
                    return str(self.elements[p_int].name)
                elif orientation == QtCore.Qt.Vertical:
                    if self.circuit.time_profile is None:
                        return str(p_int)
                    else:
                        # return pd.to_datetime(self.time_array[p_int]).strftime('%d/%m/%Y %H:%M')
                        return pd.to_datetime(self.circuit.time_profile[p_int]).strftime('%d-%m-%Y %H:%M')

        return None

    def paste_from_clipboard(self, row_idx=0, col_idx=0):
        """

        Args:
            row_idx:
            col_idx:
        """
        n = len(self.elements)
        nt = len(self.circuit.time_profile)

        if n > 0:
            profile_property = self.elements[0].properties_with_profile[self.magnitude]
            formatter = self.elements[0].editable_headers[self.magnitude][1]

            # copy to clipboard
            cb = QApplication.clipboard()
            text = cb.text(mode=cb.Clipboard)

            rows = text.split('\n')

            # gather values
            for r, row in enumerate(rows):

                values = row.split('\t')
                r2 = r + row_idx
                for c, val in enumerate(values):
                    c2 = c + col_idx
                    try:
                        val2 = formatter(val)
                        if c2 < n and r2 < nt:
                            getattr(self.elements[c2], profile_property).values[r2, 0] = val2
                        else:
                            print('Out of profile bounds')
                    except:
                        warn("could not parse '" + str(val) + "'")
        else:
            # there are no elements
            pass

    def copy_to_clipboard(self):
        """
        Copy profiles to clipboard
        """
        n = len(self.elements)

        if n > 0:
            profile_property = self.elements[0].properties_with_profile[self.magnitude]

            # gather values
            names = [None] * n
            values = [None] * n
            for c in range(n):
                names[c] = self.elements[c].name
                values[c] = getattr(self.elements[c], profile_property).values[:, 0]
            values = np.array(values).transpose().astype(str)

            # header first
            data = '\t' + '\t'.join(names) + '\n'

            # data
            for t, date in enumerate(self.circuit.time_profile):
                data += str(date) + '\t' + '\t'.join(values[t, :]) + '\n'

            # copy to clipboard
            cb = QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(data, mode=cb.Clipboard)

        else:
            # there are no elements
            pass


class EnumModel(QtCore.QAbstractListModel):
    def __init__(self, list_of_enums):
        QtCore.QAbstractListModel.__init__(self)
        self.items = list_of_enums

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.items)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid() is True:
            if role == QtCore.Qt.DisplayRole:
                return QtCore.QVariant(self.items[index.row()].value[0])
            elif role == QtCore.Qt.ItemDataRole:
                return QtCore.QVariant(self.items[index.row()].value[0])
        return QtCore.QVariant()


def get_list_model(lst, checks=False):
    """
    Pass a list to a list model
    """
    list_model = QStandardItemModel()
    if lst is not None:
        if not checks:
            for val in lst:
                # for the list model
                item = QStandardItem(str(val))
                item.setEditable(False)
                list_model.appendRow(item)
        else:
            for val in lst:
                # for the list model
                item = QStandardItem(str(val))
                item.setEditable(False)
                item.setCheckable(True)
                item.setCheckState(QtCore.Qt.Checked)
                list_model.appendRow(item)

    return list_model


def get_checked_indices(mdl: QStandardItemModel()):
    """
    Get a list of the selected indices in a QStandardItemModel
    :param mdl:
    :return:
    """
    idx = list()
    for row in range(mdl.rowCount()):
        item = mdl.item(row)
        if item.checkState() == QtCore.Qt.Checked:
            idx.append(row)

    return np.array(idx)

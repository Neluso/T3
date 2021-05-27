from numpy import array, flip
from tkinter import messagebox


def read_data(filenames):
    if filenames == None:
        messagebox.showerror('Error', 'No file selected')
        return 0, 0, True
    x_data = list()
    y_data = list()
    for filename in filenames:
        if filename == '':
            messagebox.showerror('Error', 'No file selected')
            return 0, 0, True
        try:
            fh = open(filename)
        except:
            messagebox.showerror('Error', 'Error opening ' + filename.split('/')[-1])
            continue
        data = fh.read()
        data = data.split('\n')
        x = list()
        y = list()
        for item in data:
            item = item.split(',')
            if item == ['']:
                break
            x.append(float(item[0]))
            y.append(float(item[1]))
        x_data.append(array(x))
        y_data.append(array(y))
    return array(x_data), array(y_data), False


def read_1file(file_path):
    fh = open(file_path)
    data = fh.read()
    data = data.split('\n')
    t_ref = list()
    E_ref = list()
    for item in data:
        item = item.split(',')
        if item == ['']:
            break
        t_ref.append(float(item[0]))
        E_ref.append(float(item[1]))
    return array(t_ref), array(E_ref)


def read_slow_data(file_path):  # output: time (ps), I (nA)
    fh = open(file_path)
    data = fh.read()
    data = data.split('\n')
    t_ref = list()
    E_ref = list()
    for item in data:
        item = item.split('\t')
        if item == ['']:
            break
        t_ref.append(float(item[0]))
        E_ref.append(float(item[1]))
    return array(t_ref), - flip(array(E_ref)) * 1e9


def read_from_1file(file_path):  # reads: Hz, 1, 1, cm^-1, cm^-1
    fh = open(file_path)
    data = fh.read()
    data = data.split('\n')
    freq = list()
    n_val = list()
    n_std = list()
    alpha_val = list()
    alpha_std = list()
    for item in data:
        item = item.split(',')
        if item == ['']:
            break
        freq.append(float(item[0]))
        n_val.append(float(item[1]))
        n_std.append(float(item[2]))
        alpha_val.append(float(item[3]))
        alpha_std.append(float(item[4]))
    return array(freq), array(n_val), array(n_std), array(alpha_val), array(alpha_std)
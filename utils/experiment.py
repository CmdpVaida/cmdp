import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def mkdir(path):
    path = path.strip()
    path = path.rstrip("/")
    exists = os.path.exists(path)

    if not exists:
        os.makedirs(path)
        print(path + ' created sucessful')
        return True
    else:
        print(path + ' already exists')
        return False


class ExperimentResult:
    def __init__(self, name):
        self.name = name
        self.results = dict()
        # name of {quantity_name -> {repetition_id -> [values]}}

    def append(self, result):
        """
        Adds one repetition's results
        :param result: {quantity_name -> [values]}
        :return: None
        """
        for name, value in result.items():
            if name not in self.results:
                self.results[name] = dict()
            maplen = len(self.results[name])
            self.results[name][maplen] = value

    def save(self, dir_path, relative=True):
        if relative:
            dir_path = os.getcwd() + '/' + dir_path

        if not mkdir(dir_path):
            print('WARNING !!!!!!!!! Saving files into existing directory!')

        for key in self.results:
            for i in self.results[key]:
                file_path = dir_path + '/' + key + '_' + str(i) + '.txt'
                if key != 'models':
                    np.savetxt(file_path, self.results[key][i])
                else:
                    for j in range(len(self.results[key][i])):
                        file_path = dir_path + '/' + key + '_' + str(i) + '_' + str(j) + '.txt'
                        torch.save(self.results[key][i][j].state_dict(), file_path)

    def __np_loadtxt(self, path):
        res = np.loadtxt(path)
        if len(res.shape) == 0:
            res = np.array([res])
        return res

    def load(self, path, relative=True):
        if relative:
            path = os.getcwd() + '/' + path
        exists = os.path.exists(path)

        if exists:
            for filename in os.listdir(path):
                if filename[:6] == 'models':
                    continue
                print('Loading:', filename, filename[filename.find('_') + 1:-4])
                i = int(filename[filename.find('_') + 1:-4])
                name = filename[:filename.find('_')]
                if name not in self.results:
                    self.results[name] = dict()

                cur_path = path + '/' + filename
                self.results[name][i] = self.__np_loadtxt(cur_path)
        else:
            print('Couldn\'t load from {}'.format(path))


def plotsave(dir, data, ylim=150, hor_ticks_height=None, episodes=10):
    """
    :param dir: dir to which plot the plots
    :param data: {quantity_name -> {repetition_id -> value}}
    :param hor_ticks_height: None or some float height on which to draw horizontal ticks
    :param episodes:  episodes that were seen between each tick of gathering showitems' information. normally 10
    :return:
    """

    total_steps = 100000000
    for name in data:
        for i in data[name]:
            total_steps = min(total_steps, len(data[name][i]))

    data_shortened = {name: {i: data[name][i][:total_steps] for i in data[name]} for name in data}
    data = data_shortened

    repeat_times = 0
    for name in data:
        repeat_times = len(data[name])
        break

    iterations = [i * episodes for i in range(total_steps)]

    colormap = ['firebrick', 'darkred', 'slateblue', 'darkslateblue',  'olivedrab', 'yellowgreen', 'purple', 'darkviolet', 'pink', 'lightpink']
    num = 0
    SMALL_SIZE = 12
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 14

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)
    linewidth = 3
    Alpha = 0.5

    plt.figure(figsize=(16, 9))

    for name, showitem in data.items():

        if repeat_times > 1:
            arr_showitem = np.zeros((len(showitem), len(showitem[0])))
            for i in showitem:
                arr_showitem[i] = np.array(showitem[i])

            devia = np.std(a=arr_showitem, axis=0)

            finalshow = arr_showitem.mean(axis=0).tolist()
        else:
            finalshow = np.array(showitem[0])
            devia = np.zeros(finalshow.shape)

        print('PLOTTING', iterations, finalshow)
        plt.plot(iterations, finalshow, color=colormap[num*2], linestyle='-',
                 label=str(name), linewidth=linewidth)
        plt.fill_between(iterations, (np.array(finalshow) - devia).tolist(), (np.array(finalshow) + devia).tolist(),
                        alpha=Alpha, edgecolor=colormap[num*2], facecolor=colormap[num * 2])
        num = num + 1

    if hor_ticks_height:
        plt.plot(iterations, [hor_ticks_height for _ in range(total_steps)],
                 color=colormap[num], linestyle='dashdot', linewidth=linewidth)

    plt.legend(loc='upper right')
    plt.xlabel('# of Episodes')
    plt.ylabel('Accumulated Reward')
    plt.ylim(0, ylim)
    plt.savefig(dir)
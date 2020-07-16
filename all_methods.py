import math
import numpy as np

import colorsys
from matplotlib import pyplot as plt
from matplotlib import cm, colors, colorbar

from tqdm import tqdm

import cv2

from sklearn.preprocessing import normalize 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

import pickle
from sklearn.cluster import OPTICS, cluster_optics_dbscan

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model

way = ''
def update_way(curr_way):
	way = curr_way

"""##Методы"""

def visualize_ampField(ampField, ind):
# визуализирует первые 1000 трасс набора данных с номером ind (разные сейсмические данные)

    if (ind > ampField.shape[0] - 1):
        print('There is no dataset with this number')
    current_ampField = ampField[ind]
    
    rows = current_ampField[:1000].shape[0]
    cols = current_ampField[:1000].shape[1]
    fig, axes = plt.subplots(1, 1, figsize=(rows/30, cols/30))
    
    fig.suptitle('dataset ' + str(ind), fontsize=20, x=0.1, y=0.9)
    
    normMax = max([-current_ampField.min(), current_ampField.max()])
    norm = colors.Normalize(vmin=-normMax, vmax=normMax)
    cmap = cm.seismic
    
    axes.imshow(current_ampField[:1000].T, cmap="seismic")
    plt.show()

def create_objects(var_name, ampField, num_traces, num_samples):
# вырезает объекты задаваемого разреза из поля амплитуд
# var_name - имя для сохранения
# ampField - поле амплитуд всех наборов данных (состоит из 2d массивов разного размера (разные сейсмические данные))
# num_traces - число трасс в каждом объекте
# num_samples - число дискретов в каждом объекте

    def obj_for_one_dataset(ampField, num_traces, num_samples):
    # вырезает объекты из сейсмических данных одного набора

        # размер, который будет кратен подаваемым размерам одного объекта
        obj_traces = math.floor(ampField.shape[0] / num_traces) * num_traces
        obj_samples = math.floor(ampField.shape[1] / num_samples) * num_samples
    
        # число получаемых объектов
        elem_traces = math.floor(obj_traces / num_traces)
        elem_samples = math.floor(obj_samples / num_samples)
        num_of_elem = elem_traces * elem_samples
        print("Количество элементов по трассам {0}, по дискретам {1}, всего {2}\n".format(elem_traces, elem_samples, num_of_elem))
    
        ampField_divisible = np.copy(ampField[:obj_traces, :obj_samples])
        ampField_div_sp = np.array(np.split(ampField_divisible, elem_traces, axis = 0))

        objects = np.array(np.split(ampField_div_sp[0], elem_samples, axis = 1))
        for part in ampField_div_sp[1:]:
            part1 = np.array(np.split(part, elem_samples, axis = 1))
            objects = np.concatenate((objects, part1))
        return objects
    
    # проходим по всем наборам данных
    objects = obj_for_one_dataset(ampField[0], num_traces, num_samples)
    for A in ampField[1:]:
        objects = np.concatenate((objects, obj_for_one_dataset(A, num_traces, num_samples)), axis = 0)
    
    name_file = "{0}__tr={1}_smpl={2}.npy".format(var_name, num_traces, num_samples)
    np.save(way + name_file, objects)
    
    return objects

def visualize_objects(data, random=False, title='Objects'):
# визуализирует все подаваемые объекты при random=False
# при random=True визуализирует 49 случайных
# title - заголовок

    if random == True:

        ind_for_imshow = np.random.randint(0, data.shape[0], size=49)
        obj_for_imshow = data[ind_for_imshow]

        nrows = 7
        ncols = 7

        im_height = data.shape[2]
        im_width = data.shape[1]
    
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(im_width * ncols/50, im_height * nrows/50))

        for i, ax in enumerate(axes.flat):
        
          normMax = max([-obj_for_imshow[i].min(), obj_for_imshow[i].max()])
          norm = colors.Normalize(vmin=-normMax, vmax=normMax)
          cmap = cm.seismic
        
          ax.axis('off')
        
          ax.imshow(obj_for_imshow[i].T, norm=norm, cmap=cmap)
        
        return
      
    if data.shape[0] == 1 or len(data.shape) == 2:
         
        plt.axis('off')
        
        if data.shape[0] == 1:
            normMax = max([-data[0].min(), data[0].max()])
            norm = colors.Normalize(vmin=-normMax, vmax=normMax)
            cmap = cm.seismic
          
            plt.imshow(data[0].T, norm=norm, cmap=cmap)
        else:
            if len(data.shape) == 2:
                normMax = max([-data.min(), data.max()])
                norm = colors.Normalize(vmin=-normMax, vmax=normMax)
                cmap = cm.seismic

                plt.imshow(data.T, norm=norm, cmap=cmap)
        return
        
    num_obj = data.shape[0]

    nrows = int(num_obj**0.5)
    while num_obj % nrows != 0:
        nrows += 1
    ncols = int(num_obj / nrows)
    
    #if ncols == 1 or nrows == 1: # для более удобного отображения
    #  num_obj = num_obj - 1
    #  nrows = 2
    #  ncols = int(num_obj / nrows)

    im_height = data.shape[2]
    im_width = data.shape[1]
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(im_width * ncols/50, im_height * nrows/50))
    
    fig.suptitle(title, fontsize=20, x=0.1, y=0.9)
    
    for i, ax in enumerate(axes.flat):
        
        normMax = max([-data[i].min(), data[i].max()])
        norm = colors.Normalize(vmin=-normMax, vmax=normMax)
        cmap = cm.seismic
        
        ax.axis('off')
        
        ax.imshow(data[i].T, norm=norm, cmap=cmap)

def create_features1(var_name, objects, samples_step = 2, draw = False, save = True):
# для каждого объекта сигнальные (усредняем) и CV-атрибуты
# var_name - для сохранения
# из массива объектов получает массив признаков
# samples_step - время одного дискрета в мс
# draw, save - параметры для визуализации результатов вычисления

    def feat_of_one_obj(obj):
    # вычисляет признаки для одного объекта

        names_attrs = np.array(["средняя амплитуда", "среднеквадратичная амплитуда", \
        "частота максимума спектра", "энергия спектра", "ширина спектра", "центральная частота", "доля важной энергии спектра", \
        "временная разреженность", "динамическая выраженность", "индекс полосы пропускания", "доминантная частота", \
        "максимальная длина линий Хафа", "средняя длина линий Хафа", \
        "среднее угла линии Хафа (обратная величина)", "взвешенное среднее угла линии Хафа (обратная величина)", "стандартное отклонение угла линии Хафа (обратная величина)", \
        "максимальная длина контура", "средняя длина контура", "максимальная спрямлённость", "средняя спрямлённость"])

        for_obj_attrs = dict.fromkeys(names_attrs, 0)

        num_of_traces_in_objent = obj.shape[0]
        num_of_samples_in_objent = obj.shape[1]

        # АМПЛИТУДНЫЕ
        for_obj_attrs["средняя амплитуда"] = round(np.abs(obj).sum()/num_of_traces_in_objent/num_of_samples_in_objent, 5)
        for_obj_attrs["среднеквадратичная амплитуда"] = round((obj**2).sum()**(0.5)/num_of_traces_in_objent/num_of_samples_in_objent, 6)

        # СПЕКТРАЛЬНЫЕ
        # преобразование Фурье - линейная операция, поэтому можем взять Фурье от средней трассы
        # (в отличие от АКФ, где надо считать среднее от АКФ каждой трассы)
        m_trace = obj.sum(axis = 0)/num_of_traces_in_objent
        fourier = np.fft.fft(m_trace)
        fourier = fourier[int(fourier.size/2):]
        fourier = fourier[::-1]

        f_discr = 1/(samples_step/1000) # частота дискретизации (не забыли, что samples_step в мс)
        df = f_discr/num_of_samples_in_objent # шаг по частоте (ДПФ: число дискретов образа = числу дискретов прообраза)
        faxis = np.arange(0, int(round(f_discr/2)), df) # f_discr/2 - частота Найквиста
        spectra = 2/(f_discr*num_of_samples_in_objent) * (abs(fourier))**2 # нормируем квадрат амплитудного спектра

        for_obj_attrs["частота максимума спектра"] = round(spectra.argmax() * df, 5)
        for_obj_attrs["энергия спектра"] = np.trapz(spectra[:spectra.size], x=faxis)

        if spectra.max() != 0:
            for_obj_attrs["ширина спектра"] = round(for_obj_attrs["энергия спектра"]/spectra.max(), 5)

        # Здесь считаем центральную частоту по критерию, что энергии от Fmax/2 до неё и от неё до 2*Fmax совпадают
        # или, в нашем случае, отличаются друг от друга не более, чем на 5% полной энергии
        Edelta = for_obj_attrs["энергия спектра"]*0.05

        Fc, Er, El = 0, 0, 0
        # Проходим от Fmax/2 до 2*Fmax, считая обе энергии на каждом шаге и сравнивая их
        for i in range(int(np.argmax(spectra)/2)+1, int(2*np.argmax(spectra))-1):
            Fc = i*df
            try:
                El = np.trapz(spectra[int(np.argmax(spectra)/2):i], x=np.arange(np.argmax(spectra)/2*df, Fc, df))
            except ValueError:
                El = (i - int(np.argmax(spectra)/2))*df*spectra.max()/2
            try:
                Er = np.trapz(spectra[i:int(round(2*np.argmax(spectra)))], x=np.arange(Fc+df/2, 2*np.argmax(spectra)*df, df))
            except ValueError:
                Er = (int(round(2*np.argmax(spectra))) - i)*df*spectra.max()/2
            if abs(Er - El) < Edelta:
                break

        for_obj_attrs["центральная частота"] = round(Fc, 5)
        for_obj_attrs["доля важной энергии спектра"] = round((El + Er)/for_obj_attrs["энергия спектра"], 5)

        for_obj_attrs["энергия спектра"] =round(for_obj_attrs["энергия спектра"], 5) # в двух местах до этого используется: нет смысла округлять там

        # КОРРЕЛЯЦИОННЫЕ
        acfField = np.array([])
        for trace in obj:
            temp = np.correlate(trace, trace, mode='full')
            temp = temp[int(temp.size/2):]
            acfField = np.append(acfField, temp)
        acfField = np.reshape(acfField, (num_of_traces_in_objent, num_of_samples_in_objent))

        acf_params = np.array([])
        for acf_of_trace in acfField:
            acf_max = acf_of_trace.max()
            # нахождение индексов массива, после которых значение массива меняет знак
            acf_zeros = np.diff(np.sign(acf_of_trace))
            try:
                maxtomin_index = np.where(acf_zeros == -2)[0][0]
                mintomax_index = np.where(acf_zeros == 2)[0][0]
                acf_fmin = acf_of_trace[0 : mintomax_index + 1].min()
                ind_fmin = acf_of_trace[0 : mintomax_index + 1].argmin()
                x_fall1 = maxtomin_index
                y_fall1 = acf_of_trace[x_fall1]
                x_fall2 = maxtomin_index + 1
                
                #результат очень сильно зависит от способа интерполяции, поэтому вообще не будем интерполировать
                ind_fzero = x_fall2
                acf_params = np.append(acf_params, np.array([acf_max, -acf_fmin, ind_fmin, ind_fzero,2-ind_fmin/ind_fzero,1-(-acf_fmin)/acf_max,((2-ind_fmin/ind_fzero)**2+(1-(-acf_fmin)/acf_max)**2)**0.5,1/(4*ind_fzero*samples_step/1000)]))
            except IndexError:
                acf_params = np.append(acf_params, np.array([0,0,0,0,  -1,-1,-1,-1]))

        acf_params = np.reshape(acf_params, (num_of_traces_in_objent, 8))

        acf_params_mean = acf_params.sum(axis = 0)/num_of_traces_in_objent

        for_obj_attrs["временная разреженность"] = round(acf_params_mean[4], 5)
        for_obj_attrs["динамическая выраженность"] = round(acf_params_mean[5], 5)
        for_obj_attrs["индекс полосы пропускания"] = round(acf_params_mean[6], 5)
        for_obj_attrs["доминантная частота"] = round(acf_params_mean[7], 5)

        # ЗРЕНИЕ МАШИНЫ
        # делаем изображение серым (отображаем значения амплитуд в интервал 0-255, приводим полученный массив к типу byte)
        ampField_T = obj.T
        im = (np.flip(ampField_T, axis=0) - ampField_T.min())/(ampField_T.max() - ampField_T.min())*255
        im = 255 - im
        im = im.astype(np.uint8)
        im = np.flip(im)

        # чуть-чуть заблюриваем картинку (чтобы убрать шум)
        blur = cv2.GaussianBlur(im, (3,3), 0)

        # применяем пороговый фильтр
        # (алгоритм Отсу вычисляет оптимальное значение порога по точке, равноудалённой от двух пиков гистограммы,
        # соответствующих чёрному и белому цветам)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # делаем скелетизацию, т.е. истончение белых участков изображения до линий толщиной в 1 пиксель
        thresh2 = thresh.copy()
        size = np.size(thresh2)
        skel = np.zeros(thresh.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

        done = False
        while (not done):
            eroded = cv2.erode(thresh2, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(thresh2, temp)
            skel = cv2.bitwise_or(skel, temp)
            thresh2 = eroded.copy()
            zeros = size - cv2.countNonZero(thresh2)
            if zeros == size:
                done = True

        # делаем преобразование Хафа (его оптимизированную версию Probabilistic Hough Transform), которое на выходе даёт нам все найденные прямолинейные отрезки
        # skel - подаваемое изображение
        # lines - туда запишутся тета и ро каждой задетектированной линии (пишет координаты начала и конца каждой линии)
        # 2ой аргумент - разрешение по ро в пикселях
        # 3ий - разрешение по тета в радианах
        # 4ый - порог детектирования (сколько пересечений синусоид достаточно для признания линии; количество точек в линии, чтобы та задетектировалась)
        # 5ый - минимальное количество точек, способных сформировать линию (только для HoughLinesP)
        # 6ой - максимальное расстояние между двумя точками, чтобы считать, что эти точки лежат на одной линии (только для HoughLinesP)
        linesP = cv2.HoughLinesP(skel, 30, 0.3*np.pi/180, 200, None, 2, 2)
        
        #if linesP is None:
        #    print("linesP is None")
        #else:
        #    print("!!!!!!!!!!!!")
        #plt_Hl = np.zeros(shape=[im.shape[0], im.shape[1], 3], dtype=np.uint8)

        if draw == True:
          plt_Hl = np.zeros(shape=[im.shape[0], im.shape[1], 3], dtype=np.uint8) # picture for Hough lines

        houghLenghts = []
        houghAngles = []
        # для каждого отрезка находим длину и угол и сразу рисуем их на выделенном для них фоне, цветом обозначаем угол
        # затем вычисляем среднюю и максимальную длину, а также разброс по углам
        if linesP is not None:
            for i in range(len(linesP)):

                l = linesP[i][0] # координаты начала и конца очередной линии
                length = ((l[2]-l[0])**2 + (l[3]-l[1])**2)**0.5 # длина очередной линии
                houghLenghts.append(length)

                angle = np.arctan((l[3]-l[1])/(l[2]-l[0]+0.0001)) # угол наклона очередной линии
                houghAngles.append(angle*180/math.pi) # в градусах

                if draw == True:
                  #hsv_to_rgb переводит из HSV (координаты: цветовой тон от 0 до 360 градусов (поэтому и делим на math.pi), насыщенность и яркость) в RGB (красный, зеленый, синий)
                  #задаём цвет в HSV, потому что сам по себе цвет задаётся только первой координатой
                  color = colorsys.hsv_to_rgb((angle+math.pi/2)/math.pi, 1, 1)
                  color = tuple([255*x for x in color])

                  #изображение, координаты отрезка, цвет, толщина, cv2.LINE_AA - чтобы отрезок был сглажен
                  cv2.line(plt_Hl, (l[0], l[1]), (l[2], l[3]), color, 1, cv2.LINE_AA) # система RGB (для matplotlib)

            maxHLen = max(houghLenghts)
            avgHLen = np.average(houghLenghts)
            avgHAng = np.average(houghAngles)
            avgHAngW = np.average(houghAngles, weights=np.array(houghLenghts)/max(houghLenghts))
            stdHAng = np.std(houghAngles)
            if avgHAng != 0:
                avgHAng = 1 / avgHAng
            if avgHAngW != 0:
                avgHAngW = 1 / avgHAngW
            if stdHAng != 0:
                stdHAng = 1 / stdHAng
        else:
            maxHLen = 0
            avgHLen = 0
            avgHAng = 0
            avgHAngW = 0
            stdHAng = 0

        for_obj_attrs["максимальная длина линий Хафа"] = round(maxHLen, 5)
        for_obj_attrs["средняя длина линий Хафа"] = round(avgHLen, 5)
        for_obj_attrs["среднее угла линии Хафа (обратная величина)"] = round(avgHAng, 5)
        for_obj_attrs["взвешенное среднее угла линии Хафа (обратная величина)"] = round(avgHAngW, 5)
        for_obj_attrs["стандартное отклонение угла линии Хафа (обратная величина)"] = round(stdHAng, 5)

        # ищем контуры
        contours, hierarchy = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if draw == True:
          # создаются чёрные фоновые изображения, на которых будем рисовать контуры и линии
          plt_Tr = np.zeros(shape=[im.shape[0], im.shape[1], 3], dtype=np.uint8) # for Traceability
          plt_Str = plt_Tr.copy() # for Straightness

        lengths = []            # Длины по всем контурам
        straightnesses = []     # Спрямлённости
        if contours is not None:
            for cnt in contours:
                # cv2.arcLength - длина подаваемой кривой, False - говорим, что наша кривая незамкнутая
                lengths.append(cv2.arcLength(cnt, False))

                # cv2.boundingRect - the function calculates and returns the minimal up-right bounding rectangle for the specified point set
                cntRect = cv2.boundingRect(cnt)
                # спрямлённость: cntRect[3] - высота прямоугольника, описывающего контур, cntRect[2] - его ширина
                straightnesses.append(1 - cntRect[3]/cntRect[2])

                if draw == True:
                  color1 = colorsys.hsv_to_rgb(0.75 - cntRect[3]/cntRect[2], 1, 1)
                  color = colorsys.hsv_to_rgb(cv2.arcLength(cnt, False)/num_of_traces_in_objent*2, 1, 1 - cntRect[3]/cntRect[2])

                  color = tuple([255*x for x in color])
                  color1 = tuple([255*x for x in color1])
                  
                  cv2.drawContours(plt_Tr, [cnt], -1, color, 1) # система RGB (для matplotlib)
                  cv2.drawContours(plt_Str, [cnt], -1, color1, 1) # система RGB (для matplotlib)

            maxCLen = max(lengths)
            avgCLen = np.average(lengths)
            maxStraightness = max(straightnesses)
            avgStraightness = np.average(straightnesses)

        else:
            maxCLen = 0
            avgCLen = 0
            maxStraightness = 0
            avgStraightness = 0

        for_obj_attrs["максимальная длина контура"] = round(maxCLen, 5)
        for_obj_attrs["средняя длина контура"] = round(avgCLen, 5)
        for_obj_attrs["максимальная спрямлённость"] = round(maxStraightness, 5)
        for_obj_attrs["средняя спрямлённость"] = round(avgStraightness, 5)

        if draw == True:
          visualize_objects(obj)
          plt.show()

          fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(6, 18))
          
          a1 = ax1.imshow(plt_Hl, cmap="seismic") # цветом выделяется угол наклона
          ax1.set_title("Hough")
          #fig.colorbar(a1, ax=ax1)
          a2 = ax2.imshow(plt_Tr, cmap="seismic") # цветом выделяется длина контура
          ax2.set_title("Trans")
          #fig.colorbar(fig, ax=ax2)
          a3 = ax3.imshow(plt_Str, cmap="seismic") # цветом выделяется спрямленность контура
          ax3.set_title("Straight")
          #fig.colorbar(fig, ax=ax3)

          #norm1 = colors.Normalize(vmin=0, vmax=2000)
          #norm2 = colors.Normalize(vmin=0, vmax=180)
          #cmap2 = colors.LinearSegmentedColormap.from_list('mycmap', ['Cyan','Lime','Yellow','Red','Magenta','Blue'])
          #norm3 = colors.Normalize(vmin=0, vmax=1)

          #Cb1 = colorbar.ColorbarBase(ax1, norm=norm1)
          #Cb2 = colorbar.ColorbarBase(ax2, cmap=cmap2, norm=norm2)
          #Cb3 = colorbar.ColorbarBase(ax3, norm=norm3)

          plt.show()

          print()
          for keys, values in for_obj_attrs.items():
            print(keys, ' = ', values)
          print()

        return for_obj_attrs
    
    if len(objects.shape) == 2: # если подали только один объект
      attrs = np.array(list(feat_of_one_obj(objects).values()))
      attrs = attrs[np.newaxis, :]
      return attrs

    attrs = np.array(list(feat_of_one_obj(objects[0]).values()))
    attrs = attrs[np.newaxis, :]

    for elem in tqdm(objects[1:]):     
        attrs_for_iteration = np.array(list(feat_of_one_obj(elem).values()))
        attrs_for_iteration = attrs_for_iteration[np.newaxis, :]
        attrs = np.concatenate(( attrs, attrs_for_iteration ), axis = 0)
    
    if save == True:
      name_file = "{0}__tr={1}_smpl={2}.npy".format(var_name, objects[0].shape[0], objects[0].shape[1])
      np.save(way + name_file, attrs)

    return attrs

def create_features2(var_name, objects, samples_step = 2, draw = False, save = True):
# для каждого объекта получает усреднённые сигнальные и CV-атрибуты, сигнальные для каждой трассы, сохраняет картины контуров (черно-белую и со спрямлённостью)
# var_name - для сохранения
# из массива объектов получает массив признаков
# samples_step - время одного дискрета в мс
# draw, save - параметры для визуализации результатов вычисления
# если надо подать один объект (например, второй по индексу), подавать в формате create_features2('your var_name', objects[2:3])

  def feat_of_one_obj(obj):
  # вычисляет признаки для одного объекта

          traces_in_object = obj.shape[0]
          samples_in_object = obj.shape[1]

          stand_attrs = [] # ПРИЗНАКИ

          # АМПЛИТУДНЫЕ

          # все средние по трассам (obj размером 50x100)
          all_means = np.mean( np.abs(obj), axis=1 ) # ПРИЗНАКИ
          all_stds = np.std( obj, axis=1 ) # ПРИЗНАКИ
          mean = np.mean(all_means) # ПРИЗНАКИ
          stand_attrs.append(mean)
          std = np.std(np.std(obj, axis=1)) # ПРИЗНАКИ
          stand_attrs.append(std)

          # СПЕКТРАЛЬНЫЕ
          samples_step = 2
          spectr_attrs = [] # ПРИЗНАКИ

          meanTrace = np.mean(obj, axis=0)
          obj_withMeanTrace = np.vstack((meanTrace, obj)) # первая трасса - средняя (она даёт обычные спектральные атрибуты)
          for l, trace in zip(range(obj_withMeanTrace.shape[0]), obj_withMeanTrace):

            fourier = np.fft.fft(trace)
            fourier = fourier[int(fourier.size/2):]
            fourier = fourier[::-1]

            f_discr = 1/(samples_step/1000) # частота дискретизации (не забыли, что samples_step в мс)
            df = f_discr/samples_in_object # шаг по частоте (ДПФ: число дискретов образа = числу дискретов прообраза)
            faxis = np.arange(0, int(round(f_discr/2)), df) # f_discr/2 - частота Найквиста
            spectra = 2/(f_discr*samples_in_object) * (abs(fourier))**2 # нормируем квадрат амплитудного спектра

            freq_of_max = spectra.argmax() * df
            if l == 0:
              stand_attrs.append(freq_of_max)
            else:
              spectr_attrs.append(freq_of_max) # частота максимума спектра # ПРИЗНАКИ
            
            energSpec = np.trapz(spectra[:spectra.size], x=faxis) # энергия спектра # ПРИЗНАКИ
            if l == 0:
              stand_attrs.append(energSpec) # энергия спектра
            else:
              spectr_attrs.append(energSpec)

            spW = 0
            if spectra.max() != 0:
              spW = energSpec / spectra.max() # ширина спектра # ПРИЗНАКИ

            if l == 0:
              stand_attrs.append(spW) 
            else:
              spectr_attrs.append(spW)

            # Здесь считаем центральную частоту по критерию, что энергии от Fmax/2 до неё и от неё до 2*Fmax совпадают
            # или, в нашем случае, отличаются друг от друга не более, чем на 5% полной энергии
            Edelta = energSpec*0.05

            Fc, Er, El = 0, 0, 0
            # Проходим от Fmax/2 до 2*Fmax, считая обе энергии на каждом шаге и сравнивая их
            for i in range(int(np.argmax(spectra)/2)+1, int(2*np.argmax(spectra))-1):
              Fc = i*df
              try:
                  El = np.trapz(spectra[int(np.argmax(spectra)/2):i], x=np.arange(np.argmax(spectra)/2*df, Fc, df))
              except ValueError:
                  El = (i - int(np.argmax(spectra)/2))*df*spectra.max()/2
              try:
                  Er = np.trapz(spectra[i:int(round(2*np.argmax(spectra)))], x=np.arange(Fc+df/2, 2*np.argmax(spectra)*df, df))
              except ValueError:
                  Er = (int(round(2*np.argmax(spectra))) - i)*df*spectra.max()/2
              if abs(Er - El) < Edelta:
                  break
            
            if l == 0:
              stand_attrs.append(Fc) # центральная частота # ПРИЗНАКИ
            else:
              spectr_attrs.append(Fc)
            
            if energSpec != 0:
              K = (El + Er) / energSpec
              
            if l == 0:
              stand_attrs.append(K) # доля важной энергии спектра # ПРИЗНАКИ
            else:
              spectr_attrs.append(K)
          
          # КОРРЕЛЯЦИОННЫЕ
          acfField = np.array([]) # АКФ для всех трасс
          for trace in obj:
              temp = np.correlate(trace, trace, mode='full')
              temp = temp[int(temp.size/2):]
              acfField = np.append(acfField, temp)
          acfField = np.reshape(acfField, (traces_in_object, samples_in_object))

          acf_params = np.array([]) # ПРИЗНАКИ
          for acf_of_trace in acfField:
              acf_max = acf_of_trace.max()
              # нахождение индексов массива, после которых значение массива меняет знак
              acf_zeros = np.diff(np.sign(acf_of_trace))
              try:
                  maxtomin_index = np.where(acf_zeros == -2)[0][0]
                  mintomax_index = np.where(acf_zeros == 2)[0][0]
                  acf_fmin = acf_of_trace[0 : mintomax_index + 1].min()
                  ind_fmin = acf_of_trace[0 : mintomax_index + 1].argmin()
                  x_fall1 = maxtomin_index
                  y_fall1 = acf_of_trace[x_fall1]
                  x_fall2 = maxtomin_index + 1
                  
                  #результат очень сильно зависит от способа интерполяции, поэтому вообще не будем интерполировать
                  ind_fzero = x_fall2
                  acf_params = np.append(acf_params, np.array([2-ind_fmin/ind_fzero,1-(-acf_fmin)/acf_max,((2-ind_fmin/ind_fzero)**2+(1-(-acf_fmin)/acf_max)**2)**0.5,1/(4*ind_fzero*samples_step/1000)]))
              except IndexError:
                  acf_params = np.append(acf_params, np.array([-1,-1,-1,-1]))

          acf_params_reshape = np.reshape(acf_params, (traces_in_object, 4))

          acf_params_mean = acf_params_reshape.sum(axis = 0)/traces_in_object

          # в конце добавим обычный усреднённый варинат
          stand_attrs.append(acf_params_mean[0]) # временная разреженность # ПРИЗНАКИ
          stand_attrs.append(acf_params_mean[1]) # динамическая выраженность # ПРИЗНАКИ
          stand_attrs.append(acf_params_mean[2]) # индекс полосы пропускания # ПРИЗНАКИ
          stand_attrs.append(acf_params_mean[3]) # доминантная частота # ПРИЗНАКИ

          # ЗРЕНИЕ МАШИНЫ
          # делаем изображение серым (отображаем значения амплитуд в интервал 0-255, приводим полученный массив к типу byte)
          ampField_T = obj.T
          im = (np.flip(ampField_T, axis=0) - ampField_T.min())/(ampField_T.max() - ampField_T.min())*255
          im = 255 - im
          im = im.astype(np.uint8)
          im = np.flip(im)

          # чуть-чуть заблюриваем картинку (чтобы убрать шум)
          blur = cv2.GaussianBlur(im, (3,3), 0)

          # применяем пороговый фильтр
          # (алгоритм Отсу вычисляет оптимальное значение порога по точке, равноудалённой от двух пиков гистограммы,
          # соответствующих чёрному и белому цветам)
          ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

          # делаем скелетизацию, т.е. истончение белых участков изображения до линий толщиной в 1 пиксель
          thresh2 = thresh.copy()
          size = np.size(thresh2)
          skel = np.zeros(thresh.shape, np.uint8)
          element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

          done = False
          while (not done):
              eroded = cv2.erode(thresh2, element)
              temp = cv2.dilate(eroded, element)
              temp = cv2.subtract(thresh2, temp)
              skel = cv2.bitwise_or(skel, temp)
              thresh2 = eroded.copy()
              zeros = size - cv2.countNonZero(thresh2)
              if zeros == size:
                  done = True

          # делаем преобразование Хафа (его оптимизированную версию Probabilistic Hough Transform), которое на выходе даёт нам все найденные прямолинейные отрезки
          # skel - подаваемое изображение
          # lines - туда запишутся тета и ро каждой задетектированной линии (пишет координаты начала и конца каждой линии)
          # 2ой аргумент - разрешение по ро в пикселях
          # 3ий - разрешение по тета в радианах
          # 4ый - порог детектирования (сколько пересечений синусоид достаточно для признания линии; количество точек в линии, чтобы та задетектировалась)
          # 5ый - минимальное количество точек, способных сформировать линию (только для HoughLinesP)
          # 6ой - максимальное расстояние между двумя точками, чтобы считать, что эти точки лежат на одной линии (только для HoughLinesP)
          linesP = cv2.HoughLinesP(skel, 30, 0.3*np.pi/180, 200, None, 2, 2)

          if draw == True:
            plt_Hl = np.zeros(shape=[im.shape[0], im.shape[1], 3], dtype=np.uint8) # picture for Hough lines

          houghLenghts = []
          houghAngles = []
          maxHLen = 0
          avgHLen = 0
          avgHAng = 0
          avgHAngW = 0
          stdHAng = 0
          # для каждого отрезка находим длину и угол и сразу рисуем их на выделенном для них фоне, цветом обозначаем угол
          # затем вычисляем среднюю и максимальную длину, а также разброс по углам
          if linesP is not None:
              for i in range(len(linesP)):

                  l = linesP[i][0] # координаты начала и конца очередной линии
                  length = ((l[2]-l[0])**2 + (l[3]-l[1])**2)**0.5 # длина очередной линии
                  houghLenghts.append(length)

                  angle = np.arctan((l[3]-l[1])/(l[2]-l[0]+0.0001)) # угол наклона очередной линии
                  houghAngles.append(angle*180/math.pi) # в градусах

                  if draw == True:
                    #hsv_to_rgb переводит из HSV (координаты: цветовой тон от 0 до 360 градусов (поэтому и делим на math.pi), насыщенность и яркость) в RGB (красный, зеленый, синий)
                    #задаём цвет в HSV, потому что сам по себе цвет задаётся только первой координатой
                    color = colorsys.hsv_to_rgb((angle+math.pi/2)/math.pi, 1, 1)
                    color = tuple([255*x for x in color])

                    #изображение, координаты отрезка, цвет, толщина, cv2.LINE_AA - чтобы отрезок был сглажен
                    cv2.line(plt_Hl, (l[0], l[1]), (l[2], l[3]), color, 1, cv2.LINE_AA) # система RGB (для matplotlib)

              maxHLen = max(houghLenghts)
              avgHLen = np.average(houghLenghts)
              avgHAng = np.average(houghAngles)
              avgHAngW = np.average(houghAngles, weights=np.array(houghLenghts)/max(houghLenghts))
              stdHAng = np.std(houghAngles)
              if avgHAng != 0:
                  avgHAng = 1 / avgHAng
              if avgHAngW != 0:
                  avgHAngW = 1 / avgHAngW
              if stdHAng != 0:
                  stdHAng = 1 / stdHAng

          stand_attrs.append(maxHLen) # максимальная длина линий Хафа # ПРИЗНАКИ
          stand_attrs.append(avgHLen) # средняя длина линий Хафа # ПРИЗНАКИ
          stand_attrs.append(avgHAng) # среднее угла линии Хафа (обратная величина) # ПРИЗНАКИ
          stand_attrs.append(avgHAngW) # взвешенное среднее угла линии Хафа (обратная величина) # ПРИЗНАКИ
          stand_attrs.append(stdHAng) # стандартное отклонение угла линии Хафа (обратная величина) # ПРИЗНАКИ

          # ищем контуры
          contours, hierarchy = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

          # создаются чёрные фоновые изображения, на которых будем рисовать контуры
          plt_Feat = np.zeros(shape=[im.shape[0], im.shape[1]], dtype=np.uint8) # ПРИЗНАКИ
          plt_Str = np.zeros(shape=[im.shape[0], im.shape[1], 3], dtype=np.uint8) # for Straightness # ПРИЗНАКИ
          if draw == True:
            plt_Tr = plt_Str.copy() # for Traceability

          lengths = []            # Длины по всем контурам # ПРИЗНАКИ
          straightnesses = []     # Спрямлённости # ПРИЗНАКИ
          maxCLen = 0
          avgCLen = 0
          maxStraightness = 0
          avgStraightness = 0
          if contours is not None:
              for cnt in contours:
                  # cv2.arcLength - длина подаваемой кривой, False - говорим, что наша кривая незамкнутая
                  lengths.append(cv2.arcLength(cnt, False))

                  # cv2.boundingRect - the function calculates and returns the minimal up-right bounding rectangle for the specified point set
                  cntRect = cv2.boundingRect(cnt)
                  # спрямлённость: cntRect[3] - высота прямоугольника, описывающего контур, cntRect[2] - его ширина
                  straightnesses.append(1 - cntRect[3]/cntRect[2])

                  cv2.drawContours(plt_Feat, [cnt], -1, [255,255,255], 1)

                  color1 = colorsys.hsv_to_rgb(0.75 - cntRect[3]/cntRect[2], 1, 1)
                  color1 = tuple([255*x for x in color1])
                  cv2.drawContours(plt_Str, [cnt], -1, color1, 1) # система RGB (для matplotlib)

                  if draw == True:
                    color = colorsys.hsv_to_rgb(cv2.arcLength(cnt, False)/traces_in_object*2, 1, 1 - cntRect[3]/cntRect[2])
                    color = tuple([255*x for x in color])
                    cv2.drawContours(plt_Tr, [cnt], -1, color, 1) # система RGB (для matplotlib)
                    

              maxCLen = max(lengths) # ПРИЗНАКИ
              avgCLen = np.average(lengths) # ПРИЗНАКИ
              maxStraightness = max(straightnesses) # ПРИЗНАКИ
              avgStraightness = np.average(straightnesses) # ПРИЗНАКИ

          stand_attrs.append(maxCLen) # максимальная длина контура # ПРИЗНАКИ
          stand_attrs.append(avgCLen) # средняя длина контура # ПРИЗНАКИ
          stand_attrs.append(maxStraightness) # максимальная спрямлённость # ПРИЗНАКИ
          stand_attrs.append(avgStraightness) # средняя спрямлённость # ПРИЗНАКИ
          stand_attrs.append(len(lengths)) # число найденных контуров # ПРИЗНАКИ

          if draw == True:
            visualize_objects(obj)
            plt.show()

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(6, 24))
            
            a1 = ax1.imshow(plt_Hl, cmap="seismic") # цветом выделяется угол наклона
            ax1.set_title("Hough")

            a2 = ax2.imshow(plt_Tr, cmap="seismic") # цветом выделяется длина контура
            ax2.set_title("Trans")

            a3 = ax3.imshow(plt_Str, cmap="seismic") # цветом выделяется спрямленность контура
            ax3.set_title("Straight")

            a4 = ax4.imshow(plt_Feat)
            ax4.set_title("Feat")

            plt.show()

            names_attrs = np.array(["средняя амплитуда", "среднеквадратичная амплитуда", \
            "частота максимума спектра", "энергия спектра", "ширина спектра", "центральная частота", "доля важной энергии спектра", \
            "временная разреженность", "динамическая выраженность", "индекс полосы пропускания", "доминантная частота", \
            "максимальная длина линий Хафа", "средняя длина линий Хафа", \
            "среднее угла линии Хафа (обратная величина)", "взвешенное среднее угла линии Хафа (обратная величина)", "стандартное отклонение угла линии Хафа (обратная величина)", \
            "максимальная длина контура", "средняя длина контура", "максимальная спрямлённость", "средняя спрямлённость", "число контуров"])

            for_obj_attrs = dict(zip(names_attrs, stand_attrs))

            print()
            for keys, values in for_obj_attrs.items():
              print(keys, ' = ', values)
            print()
          
          sign_feat = []
          sign_feat.extend(all_means)
          sign_feat.extend(all_stds)
          sign_feat.extend(spectr_attrs)
          sign_feat.extend(acf_params)

          cv_feat = plt_Feat.flatten()
          #cv_feat_forConv = plt_Feat # потом можно получить через reshape
          
          all_feat = []
          all_feat.extend(stand_attrs)
          all_feat.extend(sign_feat)
          all_feat.extend(cv_feat)
          
          return np.array(all_feat), np.array(plt_Feat)[np.newaxis, :], np.array(plt_Str)[np.newaxis, :]

  features2 = np.zeros(5571)
  pict_bw = np.zeros((1, 100, 50))
  pict_str = np.zeros((1, 100, 50, 3))
  for elem in tqdm(objects):
    f = feat_of_one_obj(elem)
    
    features2 = np.vstack((features2, f[0]))
    pict_bw = np.vstack((pict_bw, f[1]))
    pict_str = np.vstack((pict_str, f[2]))

  if save == True:
    name_file = "{0}.npy".format(var_name)
    np.save(way + name_file, features2[1:])
    
    name_file = "{0}_bw.npy".format(var_name)
    np.save(way + name_file, pict_bw[1:])

    name_file = "{0}_str.npy".format(var_name)
    np.save(way + name_file, pict_str[1:])

  return features2[1:], pict_bw[1:], pict_str[1:]

def reduce_pca(var_name, data, dim):
# снижает размерность признаков data до dim
# var_name - для сохранения

  pca = PCA(n_components = dim)
  data_pca = pca.fit_transform(data)
  
  name_file = '{0}.npy'.format(var_name)
  np.save(way + name_file, data_pca)

  return data_pca

def reduce_isomap(var_name, data, dim, n_neighbors):
# снижает размерность признаков data до dim
# n_neighbors - число соседей, которые нужно учитывать для каждой точки
# var_name - для сохранения

  iso = Isomap(n_components=dim, n_neighbors=n_neighbors, n_jobs=-1, p=1)
  data_iso = iso.fit_transform(data)

  name_file = '{0}__neighbors={1}.npy'.format(var_name, n_neighbors)
  np.save(way + name_file, data_iso)

  return data_iso

def reduce_autoencoder(var_name, data, dim):
# снижает размерность признаков data до dim
# var_name - для сохранения

  x_train, x_test = train_test_split(data, test_size = 0.15, random_state=32, shuffle = True) # shuffle = True - перемешиваем выборку
  
  # изменением гиперпараметров (см. Autoencoders for seismic) была подобрана следующая модель:
  # encoder
  input_attrs = Input(shape = (20,))
  x = Dense(19, activation='tanh', bias_initializer='zeros', kernel_initializer='random_uniform')(input_attrs)
  x = Dense(17, activation='tanh', bias_initializer='zeros', kernel_initializer='random_uniform')(x)
  code = Dense(dim, activation='linear')(x)
  
  # decoder
  input_code = Input(shape = (dim,))
  x = Dense(17, activation='tanh', bias_initializer='zeros', kernel_initializer='random_uniform')(input_code)
  x = Dense(19, activation='tanh', bias_initializer='zeros', kernel_initializer='random_uniform')(x)
  out_attrs = Dense(20, activation='linear')(x)

  encoder = Model(input_attrs, code, name="encoder")
  decoder = Model(input_code, out_attrs, name="decoder")
  d_ae = Model(input_attrs, decoder(encoder(input_attrs)), name="autoencoder")

  l_rate = 0.0005 # скорость обучения
  d_ae.compile(Adam(l_rate), loss='mse', metrics=['mae']) # две метрики качества модели: средняя квадратичная ошибка (по ней обучается) и средняя абсолютная

  # stop training when a monitored quantity has stopped improving
  # patience - сколько эпох терпеть не улучшения качества (здесь не уменьшения val_loss)
  # restore_best_weights=True - сохранятся веса с лучшим качеством (здесь с минимальным val_loss)
  # обычно добавляется как callback в model.fit, но мы используем scikit-learn API; чтобы не было проблем, надо добавить earlyStopping в GridSearchCV_object.fit, причём как-то хитро (см. ниже)
  earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

  epo = 2000 # возможно можно > 2000

  hist = d_ae.fit(x_train, # shape[0] у подаваемых объектов должен быть их числом
                x_train, 
                epochs=epo, # в конце каждой эпохи проводится мониторинг функции потерь (val_loss) на validation_data (эпоха - отработать все объекты (случайно выбирается в течение эпохи количество объектов = количеству объектов в тренировочной выборке))
                batch_size=70, # сколько объектов будет накоплено (в сумму функционала ошибки) для одного обновления градиента (по умолчанию 32)
                shuffle=True, # перемешивать ли тренировочные данные в конце каждой эпохи
                verbose=1, # показ progress bar
                validation_data=(x_test, x_test), # данные для контроля переобучения
                callbacks = [earlyStopping])
  
  # графики обучения
  N = np.arange(0, len(hist.history["loss"]))
  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

  ax1.plot(N, hist.history["loss"], label="train_loss")
  ax1.plot(N, hist.history["val_loss"], label="val_loss")

  ax1.set_title("Loss")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("MSE")
  ax1.legend()

  fig = plt.figure()
  ax2.plot(N, hist.history["mae"], label="train_metrics")
  ax2.plot(N, hist.history["val_mae"], label="val_metrics")

  ax2.set_title("Metrics")
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel("MAE")
  ax2.legend()

  print()
  print('loss и metrics на тестовых данных', d_ae.evaluate(x_test, x_test))

  name_file = '{0}.npy'.format(var_name)
  d_ae.save(way + name_file)

  data_autoen = d_ae.get_layer(name="encoder").predict(data)

  return data_autoen

def draw_samples_2D(data, limx=[0, 0], limy=[0, 0]):
# рисуем точки data, limx limy - масштаб по осям

  fig, ax = plt.subplots(figsize=(5, 5))
  ax.scatter(data[:, 0], data[:, 1], s=0.05, color='black')

  #plt.tick_params(axis='both', which='major', labelsize=28)
  #plt.show()
  if limx[0] != limx[1]:
    ax.set_xlim(limx[0], limx[1])
  if limy[0] != limy[1]:
    ax.set_ylim(limy[0], limy[1])

def clust_optics(var_name, data, min_samples, xi, min_cluster_size):
# кластеризация данных data через optics; параметры см. ниже
# var_name - для сохранения

  # Unlike DBSCAN, the OPTICS algorithm does not produce a strict cluster partition, 
  # but an augmented ordering of the database. To produce the cluster partition, you can use OPTICSxi (already in OPTICS method in sklearn), 
  # which is algorithm that produces a classification based on the output of OPTICS.
  # Parametr 'xi' controls directly the number of classes we will obtain.
  # cluster_optics_dbscan in sklearn is another way to produce clusters from output of OPTICS.

  # по сути главное отличие от dbscan, что optics использует range of epsilons; 
  # чем меньше epsilon, тем меньше кластеры, тем их больше; увеличивая epsilon, кластеры сливаются (так в dbscan);
  # поэтому, если работать с range of epsilons, можно получить иерархическое разбиение на кластеры

  optics = OPTICS(
  # по сути для построения графика reachability:
               min_samples=min_samples, # число соседних точек, которые сделают данную точку корневым объектом (требующихся для образования нового кластера)
               max_eps=np.inf, # максимальная окрестность рассматриваемой точки, в которой все попавшие остальные точки считаются соседями рассматриваемой точки
               metric='minkowski', # функция измерения расстояния
               p=1, # параметр для метрики Минковского (лучшая метрика для многомерных пространств)
  # работают с выходом OPTICS, чтобы извлечь кластеры (извлечь из графика reachability):
               cluster_method='xi', # метод извлечения кластеров из выхода алгоритма OPTICS (ещё можно через 'dbscan' (см. ниже))
               xi=xi, # it is relative decrease in density (для отделения кластеров друг от друга)
               min_cluster_size=min_cluster_size) # размер минимального кластера = подаваемая доля от общего числа точек или число точек
  
  optics.fit(data)


  name_file = '{0}__min_samples={1}_xi={2}_min_cluster_size={3}.pkl'.format(var_name, min_samples, xi, min_cluster_size)
  with open(way + name_file,'wb') as f:
    pickle.dump(optics, f)

  return optics

def reachability_plot(optics, labels_cluster_oder=[], limx=[0, 0], limy=[0, 0], title='', obj_num=None):
# рисуем график расстояния достижимости, раскрашиваем на нём найденные кластеры, limx limy - масштаб по осям
  
  labels = optics.labels_

  if len(labels_cluster_oder) == 0:
    num_clust = np.unique(labels).shape[0]
  else:
    num_clust = np.unique(labels_cluster_oder).shape[0]

  if num_clust > 1000:
    print('too many clusters')
    return
  
  space = np.arange(labels.shape[0])

  if len(labels_cluster_oder) == 0:
    labels_cluster_oder = labels[optics.ordering_]
  reachability_cluster_oder = optics.reachability_[optics.ordering_]

  colors_array = np.random.rand(num_clust, 3)
  colors_tuple = list(map(tuple, colors_array[:]))

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
  for cluster, color in zip(range(0, num_clust), colors_tuple):
    x_k = space[labels_cluster_oder == cluster]
    r_k = reachability_cluster_oder[labels_cluster_oder == cluster]
    ax.scatter(x_k, r_k, c=[color], s=7, alpha=1, marker='1')
    #ax.plot(x_k, r_k, c=[color], marker='1')
  ax.scatter(space[labels_cluster_oder == -1], reachability_cluster_oder[labels_cluster_oder == -1], c='black', s=0.1, alpha=0.7, marker='2')
  if type(obj_num) != type(None):
    ax.axvline(obj_num, 0,10, linewidth=1)
  #ax.plot(pace[labels_cluster_oder == -1], reachability_cluster_oder[labels_cluster_oder == -1], c='black', alpha=0.07, marker='2')
  
  plt.title(title)

  if limx[0] != limx[1]:
    ax.set_xlim(limx[0], limx[1])
  if limy[0] != limy[1]:
    ax.set_ylim(limy[0], limy[1])

def visualize_clusters(data, labels):
# отображает объекты (разрезы) по кластерам; если больше 100 объектов в кластере, то отображает случайные 100 объектов

    num_clust = np.unique(labels).shape[0] # метку -1 тоже считаем
    
    for i in range(-1, num_clust - 1):
        
        data_in_clust = data[labels == i]
        num_in_clust = data_in_clust.shape[0]
        
        if num_in_clust <= 100:
            visualize_objects(data_in_clust, title="Cluster {0} with {1} objects".format(i, num_in_clust))
        else:   
            ind_for_imshow = np.random.randint(0, num_in_clust, size=100)
            visualize_objects(data_in_clust[ind_for_imshow], title="Cluster {0} with {1} objects".format(i, num_in_clust))

def draw_clusters_2D(data, labels, limx=[0, 0], limy=[0, 0]):
# рисует точки data и раскрашивает их по кластерам labels, limx limy - масштаб по осям
# title - заголовок графика
# labels_cluster_oder - сюда подаются метки объектов, соответствующих участкам графика достижимости без впадин (метки объектов вне кластеров) (см. ячейку ниже)

  num_clust = np.unique(labels).shape[0]
  if num_clust > 1000:
    print('too many clusters')
    return

  colors_array = np.random.rand(num_clust, 3)
  colors_tuple = list(map(tuple, colors_array))

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
  for cluster, color in zip(range(0, num_clust), colors_tuple):
    cl_data = data[labels == cluster]
    ax.scatter(cl_data[:,0], cl_data[:,1], c=[color], s=5)
  ax.scatter(data[labels == -1, 0], data[labels == -1, 1], c='black', marker='1', s=0.05)

  if limx[0] != limx[1]:
    ax.set_xlim(limx[0], limx[1])
  if limy[0] != limy[1]:
    ax.set_ylim(limy[0], limy[1])

def heavy_clusters(optics):
# возвращает метки объектов на линиях между минимумами на графике достижимости (т.е. метки объектов вне кластеров) 
# в порядке их следования по графику достижимости
# (как будто эти линии кластеры)

  num_clust = np.unique(optics.labels_).shape[0]
  labels_cluster_oder = optics.labels_[optics.ordering_]

  clust_id = np.arange(num_clust - 1)

  cl_limits = [0]
  for cl_id in clust_id:
    obj_in_cl = np.where(labels_cluster_oder == cl_id)[0] # np.where возвращает tuple с массивом

    cl_start = obj_in_cl[0]
    cl_limits.append(cl_start)

    cl_end = obj_in_cl[len(obj_in_cl) - 1]
    cl_limits.append(cl_end)

  cl_limits.append(len(labels_cluster_oder)-1)
  cl_limits = np.array(cl_limits)

  cl_limits_ = cl_limits

  labels_new = np.arange(optics.labels_.shape[0])
  st = 0
  end = 1
  cl_id = 0
  while end != len(cl_limits):
    if cl_id % 2 == 0:
      labels_new[cl_limits[st]:cl_limits[end]+1] = int(cl_id/2)
    else:
      labels_new[cl_limits[st]:cl_limits[end]+1] = -1

    cl_id += 1  
    st += 1
    end += 1
    
  return labels_new
import numpy as np
from scipy.optimize import minimize
import streamlit as st
import pandas as pd

def r2 (mas_progn, mas_real):
    SSE = sum([(mas_progn[i]-mas_real[i])**2 for i in range (len(mas_real))])
    mean = sum(mas_real)/len(mas_real)
    SST = sum([(mas_progn[i]-mean)**2 for i in range (len(mas_real))])
    return 1 - (SSE/SST)

on = st.toggle("Ввести данные в файле?")
a_list=[]
if on:
    data = st.file_uploader('Добавьте файл, где целевая переменная названа y_t', type=['xlsx', 'xls'])
    if data:
        df = pd.read_excel(data.read())
        a_list = df['y_t'].tolist()
        n = len(a_list)
else:
    n = st.number_input(label = 'Введите количество данных: ', min_value = 5)
    a_s = {'Номер':[i+1 for i in range (n)], 'Значение':[None for i in range (n)]}
    a_df = pd.DataFrame(a_s)
    a = st.data_editor(a_df, column_config = {
                                            "Значение": st.column_config.NumberColumn(
                                            "Значение",
                                            width="medium",
                                            required = True,
                                            )}, disabled=['Номер'], hide_index=True)
    if not None in a['Значение'].tolist():
        a_list = a['Значение'].tolist()
    else:
        st.warning('Заполните все поля ввода значений')
if a_list!=[]:
    def f_cheb(x):  #необходимо "собрать" функцию, которую будем минимизировать, аргументом которой будет только структура портфеля x
        mas = []
        for i in range (len(a_list)):
            mas.append(abs(x[0]+x[1]*(i+1)-a_list[i]))
        return max(mas)
    x_start = np.array([0, 0]) #x_start представляет собой начальное предположение точки минимума
    result_cheb = minimize(f_cheb, x_start, method = 'SLSQP')
    x_cheb = result_cheb.x.tolist()
    change = [abs(x_cheb[0]+x_cheb[1]*(i+1)-a_list[i]) for i in range (len(a_list))]
    #st.write(change)
    max_val = 0
    max_k = 0
    for i in range (1, len(change)-3):
        if change[i] > max_val:
            max_val = change[i]
            max_k = i
    max_k+=1
    number = st.number_input(label = 'Введите ограничивающий номер: ', min_value = 3, max_value=n-2, value = max_k)
    a_1=a_list[:number-1]
    a_2=a_list[number:]
    def const(x):
        return x[0] + x[1]*(number) - a_list[number-1]
    def f_1_mnk(x):  #необходимо "собрать" функцию, которую будем минимизировать, аргументом которой будет только структура портфеля x
        summa = 0
        for i in range (len(a_1)):
            summa += (x[0]+x[1]*(i+1)-a_1[i])**2
        return summa
    def f_2_mnk(x):  #необходимо "собрать" функцию, которую будем минимизировать, аргументом которой будет только структура портфеля x
        summa = 0
        for i in range (len(a_2)):
            summa += (x[0]+x[1]*(number+i+1)-a_2[i])**2
        return summa
    def f_1_cheb(x):  #необходимо "собрать" функцию, которую будем минимизировать, аргументом которой будет только структура портфеля x
        mas = []
        for i in range (len(a_1)):
            mas.append(abs(x[0]+x[1]*(i+1)-a_1[i]))
        return max(mas)
    def f_2_cheb(x):  #необходимо "собрать" функцию, которую будем минимизировать, аргументом которой будет только структура портфеля x
        mas = []
        for i in range (len(a_2)):
             mas.append(abs(x[0]+x[1]*(number+i+1)-a_2[i]))
        return max(mas)
    constarnt = [{'type':'eq', 'fun': const}]
    x_start = np.array([0, 0]) #x_start представляет собой начальное предположение точки минимума
    
    result_1_mnk = minimize(f_1_mnk, x_start, method = 'SLSQP', constraints=constarnt)
    x_1_mnk = result_1_mnk.x.tolist() #запись значений вектора x = (x_1, x_2, ...)
    result_2_mnk = minimize(f_2_mnk, x_start, method = 'SLSQP', constraints=constarnt)
    x_2_mnk = result_2_mnk.x.tolist() #запись значений вектора x = (x_1, x_2, ...)
    
    result_1_cheb = minimize(f_1_cheb, x_start, method = 'SLSQP', constraints=constarnt)
    x_1_cheb = result_1_cheb.x.tolist() #запись значений вектора x = (x_1, x_2, ...)
    result_2_cheb = minimize(f_2_cheb, x_start, method = 'SLSQP', constraints=constarnt)
    x_2_cheb = result_2_cheb.x.tolist() #запись значений вектора x = (x_1, x_2, ...)
    st.write('Полученные значения для первой части:')
    st.write('по МНК:        а_0 =', x_1_mnk[0], ', а_1 =', x_1_mnk[1])
    st.write('уравнение: y_t =', x_1_mnk[0], '+', x_1_mnk[1], '*t')
    st.write('по Чебышеву:   а_0 =', x_1_cheb[0], ', а_1 =', x_1_cheb[1])
    st.write('уравнение: y_t =', x_1_cheb[0], '+', x_1_cheb[1], '*t')
    st.write('Полученные значения для второй части:')
    st.write('по МНК:        а_0 =', x_2_mnk[0], ', а_1 =', x_2_mnk[1])
    st.write('уравнение: y_t =', x_2_mnk[0], '+', x_2_mnk[1], '*t')
    st.write('по Чебышеву:   а_0 =', x_2_cheb[0], ', а_1 =', x_2_cheb[1])
    st.write('уравнение: y_t =', x_2_cheb[0], '+', x_2_cheb[1], '*t')
    a_s_1 = {'Номер':[i+1 for i in range (number)], 'Реальное значение': a_list[:number],
             'Прогноз по МНК':[x_1_mnk[0]+x_1_mnk[1]*(i+1) for i in range (number)],
             'Разница по МНК':[abs(x_1_mnk[0]+x_1_mnk[1]*(i+1)-a_list[:number][i]) for i in range (number)],
             'Прогноз по Чебышеву':[x_1_cheb[0]+x_1_cheb[1]*(i+1) for i in range (number)],
             'Разница по Чебышеву':[abs(x_1_cheb[0]+x_1_cheb[1]*(i+1)-a_list[:number][i]) for i in range (number)],
             }
    a_df_1 = pd.DataFrame(a_s_1)
    st.write('Реальные и прогнозные значения для первой части')
    a_1 = st.data_editor(a_df_1, disabled=['Номер', 'Реальное значение', 'Прогноз по МНК', 'Прогноз по Чебышеву', 'Разница по МНК', 'Разница по Чебышеву'],
                         hide_index=True, column_config = {
                                        "Реальное значение": st.column_config.TextColumn(),
                                        "Прогноз по МНК": st.column_config.TextColumn(),
                                        "Прогноз по Чебышеву": st.column_config.TextColumn(),
                                        "Разница по МНК": st.column_config.TextColumn(),
                                        "Разница по Чебышеву": st.column_config.TextColumn(),
                                        })
    a_s_2 = {'Номер':[i+1 for i in range (number-1, n)], 'Реальное значение': a_list[number-1:],
             'Прогноз по МНК':[x_2_mnk[0]+x_2_mnk[1]*(i+1) for i in range (number-1, n)],
             'Разница по МНК':[abs(x_2_mnk[0]+x_2_mnk[1]*(i+1)-a_list[number-1:][i-number+1]) for i in range (number-1, n)],
             'Прогноз по Чебышеву':[x_2_cheb[0]+x_2_cheb[1]*(i+1) for i in range (number-1, n)], 
             'Разница по Чебышеву':[abs(x_2_cheb[0]+x_2_cheb[1]*(i+1)-a_list[number-1:][i-number+1]) for i in range (number-1, n)],
             }
    a_df_2 = pd.DataFrame(a_s_2)
    st.write('Реальные и прогнозные значения для второй части')
    a_2 = st.data_editor(a_df_2, disabled=['Номер', 'Реальное значение', 'Прогноз по МНК', 'Прогноз по Чебышеву', 'Разница по МНК', 'Разница по Чебышеву'],
                         hide_index=True, column_config = {
                                        "Реальное значение": st.column_config.TextColumn(),
                                        "Прогноз по МНК": st.column_config.TextColumn(),
                                        "Прогноз по Чебышеву": st.column_config.TextColumn(),
                                        "Разница по МНК": st.column_config.TextColumn(),
                                        "Разница по Чебышеву": st.column_config.TextColumn(),
                                        })
    a_df_2.drop(0, inplace=True)
    df_res = pd.concat([a_df_1, a_df_2])
    st.write('Значения R^2:')
    st.write('             для МНК: ', r2(df_res["Прогноз по МНК"].tolist(), df_res["Реальное значение"].tolist()))
    st.write('             для Чебышева: ', r2(df_res["Прогноз по Чебышеву"].tolist(), df_res["Реальное значение"].tolist()))
    st.write('График прогнозируемых и реальных значений')
    st.scatter_chart(df_res[['Номер',"Реальное значение", "Прогноз по МНК", "Прогноз по Чебышеву"]], x = 'Номер' )
    st.write('График погрешностей между прогнозируемыми и реальными значениями')
    st.bar_chart(df_res[['Номер',"Разница по МНК", "Разница по Чебышеву"]], x = 'Номер' )

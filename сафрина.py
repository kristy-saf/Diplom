import numpy as np
from scipy.optimize import minimize
import streamlit as st
import pandas as pd
on = st.toggle("Ввести данные в файле?")
a_list=[]
if on:
    data = st.file_uploader('Добавьте файл, где целевая переменная названа y_t', type=['xlsx', 'xls'])
    if data:
        df = pd.read_excel(data.read())
        a_list = df['y_t'].tolist()
        n = len(a_list)
else:
    n = st.number_input(label = 'Введите количество данных: ', min_value = 1)
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
    number = st.number_input(label = 'Введите ограничивающий номер: ', min_value = 1, max_value=n, value = n // 2)
    a_1=a_list[:number-1]
    a_2=a_list[number:]
    def const(x):
        return x[0] + x[1]*(number) - a_list[number-1]
    def f_1(x):  #необходимо "собрать" функцию, которую будем минимизировать, аргументом которой будет только структура портфеля x
        summa = 0
        for i in range (len(a_1)):
            summa += (x[0]+x[1]*(i+1)-a_1[i])**2
        return summa
    def f_2(x):  #необходимо "собрать" функцию, которую будем минимизировать, аргументом которой будет только структура портфеля x
        summa = 0
        for i in range (len(a_2)):
            summa += (x[0]+x[1]*(number+i+1)-a_2[i])**2
        return summa
    constarnt = [{'type':'eq', 'fun': const}]
    x_start = np.array([0, 0]) #x_start представляет собой начальное предположение точки минимума
    result_1 = minimize(f_1, x_start, method = 'SLSQP', constraints=constarnt)
    x_1 = result_1.x.tolist() #запись значений вектора x = (x_1, x_2, ...)
    result_2 = minimize(f_2, x_start, method = 'SLSQP', constraints=constarnt)
    x_2 = result_2.x.tolist() #запись значений вектора x = (x_1, x_2, ...)
    st.write('Полученные значения для первой части: а_0=', x_1[0], ', а_1=', x_1[1])
    st.write('Полученные значения для второй части: а_0=', x_2[0], ', а_1=', x_2[1])
    a_s_1 = {'Номер':[i+1 for i in range (number)], 'Реальное значение': a_list[:number],
             'Прогнозируемое значение':[x_1[0]+x_1[1]*(i+1) for i in range (number)], 'Разница':[abs(x_1[0]+x_1[1]*(i+1)-a_list[:number][i]) for i in range (number)]}
    a_df_1 = pd.DataFrame(a_s_1)
    st.write('Реальные и прогнозные значения для первой части:')
    a_1 = st.data_editor(a_df_1, disabled=['Номер', 'Реальное значение', 'Прогнозируемое значение', 'Разница'], hide_index=True, column_config = {
                                        "Реальное значение": st.column_config.TextColumn(),
                                        "Прогнозируемое значение": st.column_config.TextColumn(),
                                        "Разница": st.column_config.TextColumn()})
    a_s_2 = {'Номер':[i+1 for i in range (number-1, n)], 'Реальное значение': a_list[number-1:],
             'Прогнозируемое значение':[x_2[0]+x_2[1]*(i+1) for i in range (number-1, n)], 'Разница':[abs(x_2[0]+x_2[1]*(i+1)-a_list[number-1:][i-number+1]) for i in range (number-1, n)]}
    a_df_2 = pd.DataFrame(a_s_2)
    st.write('Реальные и прогнозные значения для второй части:')
    a_2 = st.data_editor(a_df_2, disabled=['Номер', 'Реальное значение', 'Прогнозируемое значение', 'Разница'], hide_index=True, column_config = {
                                        "Реальное значение": st.column_config.TextColumn(),
                                        "Прогнозируемое значение": st.column_config.TextColumn(),
                                        "Разница": st.column_config.TextColumn()})
    df_res = pd.concat([a_df_1, a_df_2])
    st.write('График прогнозируемых и реальных значений')
    st.scatter_chart(df_res[['Номер',"Реальное значение", "Прогнозируемое значение"]], x = 'Номер' )
    st.write('График погрешностей между прогнозируемыми и реальными значениями')
    st.bar_chart(df_res[['Номер',"Разница"]], x = 'Номер' )

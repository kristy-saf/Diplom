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
    def f_mnk(x):
        summa = 0
        for i in range (len(a_list)):
            summa += (x[0]+x[1]*(i+1)-a_list[i])**2
        return summa
    def f_cheb(x):
        mas = []
        for i in range (len(a_list)):
            mas.append(abs(x[0]+x[1]*(i+1)-a_list[i]))
        return max(mas)
    x_start = np.array([0, 0]) #x_start представляет собой начальное предположение точки минимума
    result_cheb = minimize(f_cheb, x_start, method = 'SLSQP')
    x_cheb = result_cheb.x.tolist()
    result_mnk = minimize(f_mnk, x_start, method = 'SLSQP')
    x_mnk = result_mnk.x.tolist()
    a_s = {'Номер':[i+1 for i in range (n)], 'Реальное значение': a_list,
             'Прогноз по МНК':[x_mnk[0]+x_mnk[1]*(i+1) for i in range (n)],
             'Разница по МНК':[abs(x_mnk[0]+x_mnk[1]*(i+1)-a_list[i]) for i in range (n)],
             'Прогноз по Чебышеву':[x_cheb[0]+x_cheb[1]*(i+1) for i in range (n)],
             'Разница по Чебышеву':[abs(x_cheb[0]+x_cheb[1]*(i+1)-a_list[i]) for i in range (n)],
             }
    a_df = pd.DataFrame(a_s)
    with st.expander("Линейная аппроксимация без использования сплайнов"):
        st.write('Полученные значения:')
        st.write('по МНК:        а_0 =', x_mnk[0], ', а_1 =', x_mnk[1])
        st.write('уравнение: y_t =', x_mnk[0], '+', x_mnk[1], '*t')
        st.write('по Чебышеву:   а_0 =', x_cheb[0], ', а_1 =', x_cheb[1])
        st.write('уравнение: y_t =', x_cheb[0], '+', x_cheb[1], '*t')
        st.write('Реальные и прогнозные значения для первой части')
        a_1 = st.data_editor(a_df, disabled=['Номер', 'Реальное значение', 'Прогноз по МНК', 'Прогноз по Чебышеву', 'Разница по МНК', 'Разница по Чебышеву'],
                             hide_index=True, column_config = {
                                            "Реальное значение": st.column_config.TextColumn(),
                                            "Прогноз по МНК": st.column_config.TextColumn(),
                                            "Прогноз по Чебышеву": st.column_config.TextColumn(),
                                            "Разница по МНК": st.column_config.TextColumn(),
                                            "Разница по Чебышеву": st.column_config.TextColumn(),
                                            })
        st.write('Значения R^2:')
        st.write('             для МНК: ', r2(a_df["Прогноз по МНК"].tolist(), a_df["Реальное значение"].tolist()))
        st.write('             для Чебышева: ', r2(a_df["Прогноз по Чебышеву"].tolist(), a_df["Реальное значение"].tolist()))
        st.write('График прогнозируемых и реальных значений')
        st.scatter_chart(a_df[['Номер',"Реальное значение", "Прогноз по МНК", "Прогноз по Чебышеву"]], x = 'Номер' )
        st.write('График погрешностей между прогнозируемыми и реальными значениями')
        st.bar_chart(a_df[['Номер',"Разница по МНК", "Разница по Чебышеву"]], x = 'Номер' )




    
    change = [abs(x_cheb[0]+x_cheb[1]*(i+1)-a_list[i]) for i in range (len(a_list))]
    #st.write(change)
    max_val = 0
    max_k = 0
    for i in range (1, len(change)-3):
        if change[i] > max_val:
            max_val = change[i]
            max_k = i
    number = max_k+1
    a_1=a_list[:number]
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

    a_s_1 = {'Номер':[i+1 for i in range (number)], 'Реальное значение': a_list[:number],
             'Прогноз по МНК':[x_1_mnk[0]+x_1_mnk[1]*(i+1) for i in range (number)],
             'Разница по МНК':[abs(x_1_mnk[0]+x_1_mnk[1]*(i+1)-a_list[:number][i]) for i in range (number)],
             'Прогноз по Чебышеву':[x_1_cheb[0]+x_1_cheb[1]*(i+1) for i in range (number)],
             'Разница по Чебышеву':[abs(x_1_cheb[0]+x_1_cheb[1]*(i+1)-a_list[:number][i]) for i in range (number)],
             }
    a_df_1 = pd.DataFrame(a_s_1)
    a_s_2 = {'Номер':[i+1 for i in range (number-1, n)], 'Реальное значение': a_list[number-1:],
             'Прогноз по МНК':[x_2_mnk[0]+x_2_mnk[1]*(i+1) for i in range (number-1, n)],
             'Разница по МНК':[abs(x_2_mnk[0]+x_2_mnk[1]*(i+1)-a_list[number-1:][i-number+1]) for i in range (number-1, n)],
             'Прогноз по Чебышеву':[x_2_cheb[0]+x_2_cheb[1]*(i+1) for i in range (number-1, n)], 
             'Разница по Чебышеву':[abs(x_2_cheb[0]+x_2_cheb[1]*(i+1)-a_list[number-1:][i-number+1]) for i in range (number-1, n)],
             }
    a_df_2 = pd.DataFrame(a_s_2)
    a_df_2.drop(0, inplace=True)
    df_res = pd.concat([a_df_1, a_df_2])

    with st.expander("Минимаксный метод аппроксимации двумя линейными сплайнами"):
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
        st.write('Реальные и прогнозные значения для первой части')
        a_1 = st.data_editor(a_df_1, disabled=['Номер', 'Реальное значение', 'Прогноз по МНК', 'Прогноз по Чебышеву', 'Разница по МНК', 'Разница по Чебышеву'],
                             hide_index=True, column_config = {
                                            "Реальное значение": st.column_config.TextColumn(),
                                            "Прогноз по МНК": st.column_config.TextColumn(),
                                            "Прогноз по Чебышеву": st.column_config.TextColumn(),
                                            "Разница по МНК": st.column_config.TextColumn(),
                                            "Разница по Чебышеву": st.column_config.TextColumn(),
                                            })
        
        st.write('Реальные и прогнозные значения для второй части')
        a_2 = st.data_editor(a_df_2, disabled=['Номер', 'Реальное значение', 'Прогноз по МНК', 'Прогноз по Чебышеву', 'Разница по МНК', 'Разница по Чебышеву'],
                             hide_index=True, column_config = {
                                            "Реальное значение": st.column_config.TextColumn(),
                                            "Прогноз по МНК": st.column_config.TextColumn(),
                                            "Прогноз по Чебышеву": st.column_config.TextColumn(),
                                            "Разница по МНК": st.column_config.TextColumn(),
                                            "Разница по Чебышеву": st.column_config.TextColumn(),
                                            })
        st.write('Значения R^2:')
        st.write('             для МНК: ', r2(df_res["Прогноз по МНК"].tolist(), df_res["Реальное значение"].tolist()))
        st.write('             для Чебышева: ', r2(df_res["Прогноз по Чебышеву"].tolist(), df_res["Реальное значение"].tolist()))
        st.write('График прогнозируемых и реальных значений')
        st.scatter_chart(df_res[['Номер',"Реальное значение", "Прогноз по МНК", "Прогноз по Чебышеву"]], x = 'Номер' )
        st.write('График погрешностей между прогнозируемыми и реальными значениями')
        st.bar_chart(df_res[['Номер',"Разница по МНК", "Разница по Чебышеву"]], x = 'Номер' )







    max_val = 0
    max_val_2 = 0
    max_k = 0
    max_k_2 = 0
    for i in range (1, len(change)-3):
        if change[i] > max_val:
            max_val_2, max_k_2 = max_val, max_k
            max_val = change[i]
            max_k = i
        elif change[i] > max_val_2:
            max_val_2 = change[i]
            max_k_2 = i
    max_k+=1
    max_k_2+=1
    if max_k > max_k_2:
        a_1 = a_list[:max_k_2]
        a_2 = a_list[max_k_2:max_k]
        a_3 = a_list[max_k:]
        number1 = max_k_2
        number2 = max_k
    else:
        a_1 = a_list[:max_k]
        a_2 = a_list[max_k:max_k_2]
        a_3 = a_list[max_k_2:]
        number1 = max_k
        number2 = max_k_2

    def const1(x):
        return x[0] + x[1]*(number1) - a_list[number1-1]
    def const2(x):
        return x[0] + x[1]*(number2) - a_list[number2-1]
    def f_1_mnk(x):
        summa = 0
        for i in range (len(a_1)):
            summa += (x[0]+x[1]*(i+1)-a_1[i])**2
        return summa
    def f_2_mnk(x):
        summa = 0
        for i in range (len(a_2)):
            summa += (x[0]+x[1]*(number1+i+1)-a_2[i])**2
        return summa
    def f_3_mnk(x):
        summa = 0
        for i in range (len(a_3)):
            summa += (x[0]+x[1]*(number2+i+1)-a_3[i])**2
        return summa
    def f_1_cheb(x):
        mas = []
        for i in range (len(a_1)):
            mas.append(abs(x[0]+x[1]*(i+1)-a_1[i]))
        return max(mas)
    def f_2_cheb(x):
        mas = []
        for i in range (len(a_2)):
             mas.append(abs(x[0]+x[1]*(number1+i+1)-a_2[i]))
        return max(mas)
    def f_3_cheb(x):
        mas = []
        for i in range (len(a_3)):
             mas.append(abs(x[0]+x[1]*(number2+i+1)-a_3[i]))
        return max(mas)
    constarnt1 = [{'type':'eq', 'fun': const1}]
    constarnt2 = [{'type':'eq', 'fun': const1}, {'type':'eq', 'fun': const2}]
    constarnt3 = [{'type':'eq', 'fun': const2}]
    x_start = np.array([0, 0]) 
    
    result_1_mnk = minimize(f_1_mnk, x_start, method = 'SLSQP', constraints=constarnt1)
    x_1_mnk = result_1_mnk.x.tolist()
    result_2_mnk = minimize(f_2_mnk, x_start, method = 'SLSQP', constraints=constarnt2)
    x_2_mnk = result_2_mnk.x.tolist()
    result_3_mnk = minimize(f_3_mnk, x_start, method = 'SLSQP', constraints=constarnt3)
    x_3_mnk = result_3_mnk.x.tolist()
    
    result_1_cheb = minimize(f_1_cheb, x_start, method = 'SLSQP', constraints=constarnt1)
    x_1_cheb = result_1_cheb.x.tolist()
    result_2_cheb = minimize(f_2_cheb, x_start, method = 'SLSQP', constraints=constarnt2)
    x_2_cheb = result_2_cheb.x.tolist()
    result_3_cheb = minimize(f_3_cheb, x_start, method = 'SLSQP', constraints=constarnt3)
    x_3_cheb = result_3_cheb.x.tolist()

    a_s_1 = {'Номер':[i+1 for i in range (number1)], 'Реальное значение': a_list[:number1],
             'Прогноз по МНК':[x_1_mnk[0]+x_1_mnk[1]*(i+1) for i in range (number1)],
             'Разница по МНК':[abs(x_1_mnk[0]+x_1_mnk[1]*(i+1)-a_list[:number1][i]) for i in range (number1)],
             'Прогноз по Чебышеву':[x_1_cheb[0]+x_1_cheb[1]*(i+1) for i in range (number1)],
             'Разница по Чебышеву':[abs(x_1_cheb[0]+x_1_cheb[1]*(i+1)-a_list[:number1][i]) for i in range (number1)],
             }
    a_df_1 = pd.DataFrame(a_s_1)
    a_s_2 = {'Номер':[i+1 for i in range (number1-1, number2)], 'Реальное значение': a_list[number1-1:number2],
             'Прогноз по МНК':[x_2_mnk[0]+x_2_mnk[1]*(i+1) for i in range (number1-1, number2)],
             'Разница по МНК':[abs(x_2_mnk[0]+x_2_mnk[1]*(i+1)-a_list[number1-1:number2][i-number1+1]) for i in range (number1-1, number2)],
             'Прогноз по Чебышеву':[x_2_cheb[0]+x_2_cheb[1]*(i+1) for i in range (number1-1, number2)], 
             'Разница по Чебышеву':[abs(x_2_cheb[0]+x_2_cheb[1]*(i+1)-a_list[number1-1:number2][i-number1+1]) for i in range (number1-1, number2)],
             }
    a_df_2 = pd.DataFrame(a_s_2)
    a_s_3 = {'Номер':[i+1 for i in range (number2-1, n)], 'Реальное значение': a_list[number2-1:],
             'Прогноз по МНК':[x_3_mnk[0]+x_3_mnk[1]*(i+1) for i in range (number2-1, n)],
             'Разница по МНК':[abs(x_3_mnk[0]+x_3_mnk[1]*(i+1)-a_list[number2-1:][i-number2+1]) for i in range (number2-1, n)],
             'Прогноз по Чебышеву':[x_3_cheb[0]+x_3_cheb[1]*(i+1) for i in range (number2-1, n)], 
             'Разница по Чебышеву':[abs(x_3_cheb[0]+x_3_cheb[1]*(i+1)-a_list[number2-1:][i-number2+1]) for i in range (number2-1, n)],
             }
    a_df_3 = pd.DataFrame(a_s_3)
    a_df_2.drop(0, inplace=True)
    a_df_3.drop(0, inplace=True)
    df_res = pd.concat([a_df_1, a_df_2, a_df_3])

    with st.expander("Минимаксный метод аппроксимации тремя линейными сплайнами"):
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
        st.write('Полученные значения для третьей части:')
        st.write('по МНК:        а_0 =', x_3_mnk[0], ', а_1 =', x_3_mnk[1])
        st.write('уравнение: y_t =', x_3_mnk[0], '+', x_3_mnk[1], '*t')
        st.write('по Чебышеву:   а_0 =', x_3_cheb[0], ', а_1 =', x_3_cheb[1])
        st.write('уравнение: y_t =', x_3_cheb[0], '+', x_3_cheb[1], '*t')
        st.write('Реальные и прогнозные значения для первой части')
        a_1 = st.data_editor(a_df_1, disabled=['Номер', 'Реальное значение', 'Прогноз по МНК', 'Прогноз по Чебышеву', 'Разница по МНК', 'Разница по Чебышеву'],
                             hide_index=True, column_config = {
                                            "Реальное значение": st.column_config.TextColumn(),
                                            "Прогноз по МНК": st.column_config.TextColumn(),
                                            "Прогноз по Чебышеву": st.column_config.TextColumn(),
                                            "Разница по МНК": st.column_config.TextColumn(),
                                            "Разница по Чебышеву": st.column_config.TextColumn(),
                                            }, key='a_1')
        
        st.write('Реальные и прогнозные значения для второй части')
        a_2 = st.data_editor(a_df_2, disabled=['Номер', 'Реальное значение', 'Прогноз по МНК', 'Прогноз по Чебышеву', 'Разница по МНК', 'Разница по Чебышеву'],
                             hide_index=True, column_config = {
                                            "Реальное значение": st.column_config.TextColumn(),
                                            "Прогноз по МНК": st.column_config.TextColumn(),
                                            "Прогноз по Чебышеву": st.column_config.TextColumn(),
                                            "Разница по МНК": st.column_config.TextColumn(),
                                            "Разница по Чебышеву": st.column_config.TextColumn(),
                                            }, key='a_2')
        st.write('Реальные и прогнозные значения для третьей части')
        a_3 = st.data_editor(a_df_3, disabled=['Номер', 'Реальное значение', 'Прогноз по МНК', 'Прогноз по Чебышеву', 'Разница по МНК', 'Разница по Чебышеву'],
                             hide_index=True, column_config = {
                                            "Реальное значение": st.column_config.TextColumn(),
                                            "Прогноз по МНК": st.column_config.TextColumn(),
                                            "Прогноз по Чебышеву": st.column_config.TextColumn(),
                                            "Разница по МНК": st.column_config.TextColumn(),
                                            "Разница по Чебышеву": st.column_config.TextColumn(),
                                            }, key='a_3')
        st.write('Значения R^2:')
        st.write('             для МНК: ', r2(df_res["Прогноз по МНК"].tolist(), df_res["Реальное значение"].tolist()))
        st.write('             для Чебышева: ', r2(df_res["Прогноз по Чебышеву"].tolist(), df_res["Реальное значение"].tolist()))
        st.write('График прогнозируемых и реальных значений')
        st.scatter_chart(df_res[['Номер',"Реальное значение", "Прогноз по МНК", "Прогноз по Чебышеву"]], x = 'Номер' )
        st.write('График погрешностей между прогнозируемыми и реальными значениями')
        st.bar_chart(df_res[['Номер',"Разница по МНК", "Разница по Чебышеву"]], x = 'Номер' )
    





    r2_mnk_best, r2_cheb_best = -10000000, -10000000
    x_1_mnk_best, x_2_mnk_best, x_1_cheb_best, x_2_cheb_best = [], [], [], []
    df_res_mnk_best, df_res_cheb_best = pd.DataFrame(), pd.DataFrame()
    for number in range (2, len(change)-2):
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

        a_s_1 = {'Номер':[i+1 for i in range (number)], 'Реальное значение': a_list[:number],
                 'Прогноз по МНК':[x_1_mnk[0]+x_1_mnk[1]*(i+1) for i in range (number)],
                 'Разница по МНК':[abs(x_1_mnk[0]+x_1_mnk[1]*(i+1)-a_list[:number][i]) for i in range (number)],
                 'Прогноз по Чебышеву':[x_1_cheb[0]+x_1_cheb[1]*(i+1) for i in range (number)],
                 'Разница по Чебышеву':[abs(x_1_cheb[0]+x_1_cheb[1]*(i+1)-a_list[:number][i]) for i in range (number)],
                 }
        a_df_1 = pd.DataFrame(a_s_1)
        a_s_2 = {'Номер':[i+1 for i in range (number-1, n)], 'Реальное значение': a_list[number-1:],
                 'Прогноз по МНК':[x_2_mnk[0]+x_2_mnk[1]*(i+1) for i in range (number-1, n)],
                 'Разница по МНК':[abs(x_2_mnk[0]+x_2_mnk[1]*(i+1)-a_list[number-1:][i-number+1]) for i in range (number-1, n)],
                 'Прогноз по Чебышеву':[x_2_cheb[0]+x_2_cheb[1]*(i+1) for i in range (number-1, n)], 
                 'Разница по Чебышеву':[abs(x_2_cheb[0]+x_2_cheb[1]*(i+1)-a_list[number-1:][i-number+1]) for i in range (number-1, n)],
                 }
        a_df_2 = pd.DataFrame(a_s_2)
        a_df_2.drop(0, inplace=True)
        df_res = pd.concat([a_df_1, a_df_2])
        if r2(df_res["Прогноз по МНК"].tolist(), df_res["Реальное значение"].tolist()) > r2_mnk_best:
            r2_mnk_best = r2(df_res["Прогноз по МНК"].tolist(), df_res["Реальное значение"].tolist())
            x_1_mnk_best, x_2_mnk_best = x_1_mnk, x_2_mnk
            df_res_mnk_best = df_res[['Номер', 'Реальное значение', 'Прогноз по МНК', 'Разница по МНК']]
        if r2(df_res["Прогноз по Чебышеву"].tolist(), df_res["Реальное значение"].tolist()) > r2_cheb_best:
            r2_cheb_best = r2(df_res["Прогноз по Чебышеву"].tolist(), df_res["Реальное значение"].tolist())
            x_1_cheb_best, x_2_cheb_best = x_1_cheb, x_2_cheb
            df_res_cheb_best = df_res[['Номер', 'Прогноз по Чебышеву', 'Разница по Чебышеву']]
        
            
    df_res = pd.merge(df_res_mnk_best, df_res_cheb_best, how='outer')
    with st.expander("Аппроксимация двумя линейными сплайнами с наилучшими показателями коэффициентов детерминации (R^2)"):
        st.write('Полученные значения для первой части:')
        st.write('по МНК:        а_0 =', x_1_mnk_best[0], ', а_1 =', x_1_mnk_best[1])
        st.write('уравнение: y_t =', x_1_mnk_best[0], '+', x_1_mnk_best[1], '*t')
        st.write('по Чебышеву:   а_0 =', x_1_cheb_best[0], ', а_1 =', x_1_cheb_best[1])
        st.write('уравнение: y_t =', x_1_cheb_best[0], '+', x_1_cheb_best[1], '*t')
        st.write('Полученные значения для второй части:')
        st.write('по МНК:        а_0 =', x_2_mnk_best[0], ', а_1 =', x_2_mnk_best[1])
        st.write('уравнение: y_t =', x_2_mnk_best[0], '+', x_2_mnk_best[1], '*t')
        st.write('по Чебышеву:   а_0 =', x_2_cheb_best[0], ', а_1 =', x_2_cheb_best[1])
        st.write('уравнение: y_t =', x_2_cheb_best[0], '+', x_2_cheb_best[1], '*t')
        st.write('Реальные и прогнозные значения')
        a = st.data_editor(df_res, disabled=['Номер', 'Реальное значение', 'Прогноз по МНК', 'Прогноз по Чебышеву', 'Разница по МНК', 'Разница по Чебышеву'],
                             hide_index=True, column_config = {
                                            "Реальное значение": st.column_config.TextColumn(),
                                            "Прогноз по МНК": st.column_config.TextColumn(),
                                            "Прогноз по Чебышеву": st.column_config.TextColumn(),
                                            "Разница по МНК": st.column_config.TextColumn(),
                                            "Разница по Чебышеву": st.column_config.TextColumn(),
                                            })
        st.write('Значения R^2:')
        st.write('             для МНК: ', r2(df_res["Прогноз по МНК"].tolist(), df_res["Реальное значение"].tolist()))
        st.write('             для Чебышева: ', r2(df_res["Прогноз по Чебышеву"].tolist(), df_res["Реальное значение"].tolist()))
        st.write('График прогнозируемых и реальных значений')
        st.scatter_chart(df_res[['Номер',"Реальное значение", "Прогноз по МНК", "Прогноз по Чебышеву"]], x = 'Номер' )
        st.write('График погрешностей между прогнозируемыми и реальными значениями')
        st.bar_chart(df_res[['Номер',"Разница по МНК", "Разница по Чебышеву"]], x = 'Номер' )

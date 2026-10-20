import numpy as np
import pandas as pd

import sys
sys.path.append('//')

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

# результат матча как значение
score_dct={'Win': 1, 'Lose': 0, 'Draw': 0.5,'Unknown': 0}

# результат матча как критерий
score_label_dct={'Win': '2.Win', 'Lose': '0.Lose', 'Draw': '1.Draw'}



# функция для форматирования датасета для требуемого формата
def formatting(df, gh, ga):
    # по умолчанию дата матча передается как строка 
    df['Date']=df['Date'].astype('datetime64[ns]')
    
    # определяем результат матча для домашней и гостевой команд, сравнивая количество голов
    df['home_result']=np.where((df[gh]>df[ga]), 'Win', 
                                    np.where((df[gh]<df[ga]), 'Lose', 'Draw'))
    
    df['away_result']=np.where((df[gh]>df[ga]), 'Lose', 
                                    np.where((df[gh]<df[ga]), 'Win', 'Draw'))
    
    df=df.rename(columns={
                        gh: 'home_score',
                        ga: 'away_score',
                        'home': 'home_team',
                        'away': 'away_team'})

    return df



# функция для создание отдельных датасетов команд
def team_table(df):
    # переименовываем правильно столбцы
    df_home=df.copy()
    df_home['date'] = df_home['Date']
    df_home['team'] = df_home['HomeTeam']
    df_home['opponent'] = df_home['AwayTeam']
    df_home['result'] = df_home['home_result']
    df_home['team_score'] = df_home['home_score']
    df_home['opponent_score'] = df_home['away_score']
    df_home['venue'] = 'Home'
    df_home['team_id'] = df_home['team_id']
    df_home['opponent_team_id'] = df_home['opponent_team_id']

    df_home = df_home[['match_id', 'date','team', 'opponent', 'venue', 'result', 'team_score', 'opponent_score','team_id', 'opponent_team_id']]
    
    # симметричный датасет для гостевой команды
    df_away=df.copy()
    
    df_away['date'] = df_away['Date']
    df_away['team'] = df_away['AwayTeam']
    df_away['opponent'] = df_away['HomeTeam']
    df_away['result'] = df_away['away_result']
    df_away['team_score'] = df_away['away_score']
    df_away['opponent_score'] = df_away['home_score']
    df_away['venue'] = 'Away'
    df_away['team_id'] = df_away['opponent_team_id']
    df_away['opponent_team_id'] = df_away['team_id']
    
    df_away = df_away[['match_id', 'date','team', 'opponent', 'venue', 'result', 'team_score', 'opponent_score','team_id', 'opponent_team_id']]

    
    # объединяем оба датасета
    team_df = pd.concat([df_home, df_away])
    team_df = team_df.sort_values(by = 'date', ascending = True)
    team_df = team_df.reset_index()
    
    team_df['game_number'] = team_df.sort_values(['date'], ascending = [True]).groupby(['team']).cumcount() + 1
    
    team_df['id'] = team_df.index
    
    # добавим некоторые данные
    # очки за матчи
    team_df['points'] = np.where(team_df['result']=='Win', 3,
                                        np.where(team_df['result']=='Draw', 1, 0))
    
    # результат
    team_df['score'] = np.where(team_df['result']=='Win', 1,
                                        np.where(team_df['result']=='Draw', 0.5, 0))
    
    # boolean столбец для быстрого определения победа/поражение
    result_ohe = pd.get_dummies(team_df['result'], prefix='is')
    team_df = pd.concat([team_df, result_ohe], axis=1)

    # рассортируем матчи по временной ленте
    team_df = team_df.sort_values(by=['date', 'game_number', 'venue'], ascending=[True, True, False])

    team_df['is_dry_match'] = np.where(team_df['opponent_score']==0, 1, 0)

    # суммарные значения
    team_df['wins'] = team_df.sort_values(by=['date'], ascending=True).groupby(['team'])['is_Win'].cumsum()
    
    team_df['draws'] = team_df.sort_values(by=['date'], ascending=True).groupby(['team'])['is_Draw'].cumsum()
    
    team_df['loses'] = team_df.sort_values(by=['date'], ascending=True).groupby(['team'])['is_Lose'].cumsum()

    team_df['scored'] = team_df.sort_values(by=['date'], ascending=True).groupby(['team'])['team_score'].cumsum()
    
    team_df['mised'] = team_df.sort_values(by=['date'], ascending=True).groupby(['team'])['opponent_score'].cumsum()
    
    return team_df



# функция, высчитывающая шанс победы той или иной команды на основе рейтинга Эло
def elo_prob(rw, rb):
    
    rw=float(rw)
    rb=float(rb)
    res=1/(1+np.power(10, (rb-rw)/400))
    
    return res



# функция, изменяющая рейтинг Эло после матча
def elo_rating_changes(rating, opponent_rating, score, games):
    
    K = 20
    expectation=elo_prob(rating, opponent_rating)
    new_rating=rating+K*(score-expectation)
    
    return np.round(new_rating,2)



# функция подсчета рейтинга Эло по серии матчей
def count_elo_rating_series(team_ser, opponent_ser, rating_ser, opponent_rating_ser, score_ser, game_number_ser):
    # словарь в котором хранится просчитанный рейтинг Эло каждой команды
    current_rating_dict={}
    # списки для результата
    rating_lst = []
    opponent_rating_lst = []
    # если команда NaN
    default_rating_value = 2400

    # daraframe withought details
    base_stat_df = pd.concat([team_ser, rating_ser], axis=1)
    base_stat_df.drop_duplicates(inplace=True)  
    
    current_rating_dict=dict(zip(base_stat_df[base_stat_df.columns[0]], base_stat_df[base_stat_df.columns[1]]))
    
    
    for i in range(len(team_ser)):
        # получаем рейтинг Эло соответствующей команды
        try:
            r1 = current_rating_dict[team_ser.values[i]]
        except:
            r1 = default_rating_value
        # а также для соперника   
        try:
            r2 = current_rating_dict[opponent_ser.values[i]]
        except:
            r2 = default_rating_value
        
        # рассчитываем рейтинг после матча между командами
        new_rating=elo_rating_changes(r1, r2, score_ser.values[i], game_number_ser.values[i])

        new_opponent_rating = r2 - (new_rating - r1)
        
        rating_lst.append(new_rating)
        opponent_rating_lst.append(new_opponent_rating)
        
        if (i % 2) == 1:
            current_rating_dict.update({team_ser.values[i]: new_rating})
            current_rating_dict.update({opponent_ser.values[i]: new_opponent_rating})

    return current_rating_dict, rating_lst, opponent_rating_lst


# функция для добавления данных с рейтингом Эло в датасет
def count_elo_rating_df(df):
    # исходный рейтинг всех команд
    start_ratings = pd.read_csv('start_ratings.csv')
    start_ratings['start_rating'] = start_ratings['start_rating'].fillna(2286)
    
    df = df.merge(start_ratings, 'left', left_on='team_id', right_on='id', suffixes=('', '_rating'))
    
    df = df.merge(start_ratings, 'left', left_on='opponent_team_id', right_on='id', suffixes=('', '_opponent'))
    
    df = df.drop(columns=['team_rating', 'id_rating', 'team_opponent', 'id_opponent'])
    
    df['start_rating'] = df['start_rating'].fillna(2286)
    df['start_rating_opponent'] = df['start_rating_opponent'].fillna(2286)
    
    # снизим рейтинг команд из вторых дивизионов
    df['rating'] = np.where(df['is_high_league'] == 0, df['start_rating'] - 200, df['start_rating'])
    
    df['opponent_rating'] = np.where(df['is_high_league'] == 0, df['start_rating_opponent'] - 200, df['start_rating_opponent'])
    
    df = df.drop(columns=['start_rating', 'start_rating_opponent'])

    # рассчитываем ретйинг для команд исходя из матчей
    current_rating_dict, rating_lst, opponent_rating_lst = count_elo_rating_series(
                        df['team'],
                        df['opponent'],
                        df['rating'], 
                        df['opponent_rating'], 
                        df['score'], 
                        df['game_number']
                    )
    # добавляем рейтинг к датасету
    df['rating'] = rating_lst
    df['opponent_rating'] = opponent_rating_lst
    df['rating_difference'] = df['rating'] - df['opponent_rating']
    
    df['elo_propability']=df[['rating', 'opponent_rating']].apply(lambda x: np.round(elo_prob(x[0], x[1]),2), axis=1)
    
    return df, current_rating_dict


# функция для подсчета статистики персональных встреч команд
def calc_pers_stat(df):
    # словарь, ключи в котором являются матчами между определенными командами
    curr_stat_dct = {}
    lst = []
    n_lst = []
    res_lst = []
    
    for i in range(len(df)):
        team = df['team'].values[i]
        opponent = df['opponent'].values[i]
        venue = df['venue'].values[i]
        
        key = team + '-' + opponent
        
        wins = df['is_Win'].values[i]
        draws = df['is_Draw'].values[i]
        loses = df['is_Lose'].values[i]
        
        scored = df['team_score'].values[i]
        missed = df['opponent_score'].values[i]
        
        if venue == 'Home':
                    wins_home = df['is_Win'].values[i]
                    draws_home = df['is_Draw'].values[i]
                    loses_home = df['is_Lose'].values[i]
                    scored_home = df['team_score'].values[i]
                    missed_home = df['opponent_score'].values[i]
        else:
                    wins_home = 0
                    draws_home = 0
                    loses_home = 0
                    scored_home = 0
                    missed_home = 0
                    
        if venue == 'Away':
                    wins_away = df['is_Win'].values[i]
                    draws_away = df['is_Draw'].values[i]
                    loses_away = df['is_Lose'].values[i]
                    scored_away = df['team_score'].values[i]
                    missed_away = df['opponent_score'].values[i]
        else:
                    wins_away = 0
                    draws_away = 0
                    loses_away = 0
                    scored_away = 0
                    missed_away = 0
                
        try:
            lst = [
                curr_stat_dct[key][0] + wins,
                curr_stat_dct[key][1] + draws,
                curr_stat_dct[key][2] + loses,
                curr_stat_dct[key][3] + scored,
                curr_stat_dct[key][4] + missed,
                
                curr_stat_dct[key][5] + wins_home,
                curr_stat_dct[key][6] + draws_home,
                curr_stat_dct[key][7] + loses_home,
                curr_stat_dct[key][8] + scored_home,
                curr_stat_dct[key][9] + missed_home,
                
                curr_stat_dct[key][10] + wins_away,
                curr_stat_dct[key][11] + draws_away,
                curr_stat_dct[key][12] + loses_away,
                curr_stat_dct[key][13] + scored_away,
                curr_stat_dct[key][14] + missed_away,
                  ]
        except:
            lst = [
                wins,
                draws,
                loses,
                scored,
                missed,
                
                wins_home,
                draws_home,
                loses_home,
                scored_home,
                missed_home,
                
                wins_away,
                draws_away,
                loses_away,
                scored_away,
                missed_away,
                  ]
            
        curr_stat_dct.update({key : lst})
        res_lst.append(lst)

    return curr_stat_dct, res_lst



# функция со статистикой личных встреч на основе матч дома/в гостях
def pers_stat_venue(pers_stat_df):
    pers_stat_df.columns = ['wins','draws','loses','scored','missed',
                        'wins_home','draws_home','loses_home','scored_home','missed_home',
                        'wins_away','draws_away','loses_away','scored_away','missed_away',
                        'id']

    # статистика с домашних игр
    pers_stat_df_h = pers_stat_df[['id','wins','draws','loses','scored','missed',
                                       'wins_home','draws_home','loses_home','scored_home','missed_home']].copy()
    
    pers_stat_df_h.columns = ['id','wins_personal_meet','draws_personal_meet','loses_personal_meet','scored_personal_meet','missed_personal_meet',
                                    'wins_venue','draws_venue','loses_venue','scored_venue','missed_venue']
    
    pers_stat_df_h['venue'] = 'Home'

    # статистика с гостевых игр
    pers_stat_df_a = pers_stat_df[['id','wins','draws','loses','scored','missed',
                                        'wins_away','draws_away','loses_away','scored_away','missed_away']].copy()
    
    pers_stat_df_a.columns = ['id','wins_personal_meet','draws_personal_meet','loses_personal_meet','scored_personal_meet','missed_personal_meet',
                                    'wins_venue','draws_venue','loses_venue','scored_venue','missed_venue']
    
    pers_stat_df_a['venue'] = 'Away'

    # объединим результаты с отметкой: где была игра (Home/Away)
    pers_stat_df_venue = pd.concat([pers_stat_df_h, pers_stat_df_a])
    
    pers_stat_df_venue['result_difference_personal_meet'] = pers_stat_df_venue['wins_personal_meet']  - pers_stat_df_venue['loses_personal_meet']
    
    pers_stat_df_venue['result_difference_venue'] = pers_stat_df_venue['wins_venue'] - pers_stat_df_venue['loses_venue']
    
    pers_stat_df_venue['games_personal_meet'] = (pers_stat_df_venue['wins_personal_meet']
                                        + pers_stat_df_venue['loses_personal_meet'] + pers_stat_df_venue['draws_personal_meet'])

    
    pers_stat_df_venue['games_venue'] = pers_stat_df_venue['wins_venue']  + pers_stat_df_venue['loses_venue']  + pers_stat_df_venue['draws_venue']

    return pers_stat_df_venue


# функция для создания датасета, содержащего данных личных встреч
def calc_pers_stat_df(df):
    
    curr_stat_dct, res_lst = calc_pers_stat(df)
    pers_stat_df = pd.DataFrame(res_lst)
    pers_stat_df['id'] = df['id'].values
    
    pers_stat_df_venue = pers_stat_venue(pers_stat_df)
    
    df = df.merge(pers_stat_df_venue, 'left', on=['id', 'venue'], suffixes=('', '_personal_meet'))
    
    return df, curr_stat_dct


    
# функция прогноза матча на основе рейтинга Эло    
def elo_prob_predict(prob, draw_share):

    try:
        draw_balance=0.5
        if prob>draw_balance+draw_share:
            return 'Win'
        elif prob<draw_balance-draw_share:
            return 'Lose'
        else:
            return 'Draw'
    except:
        return 'Unknown'


# функция простого прогноза 
def simple_predict(team, opponent, venue):
    return 'Win'

# функция рандомного прогноза
def random_predict(team, opponent, seed):
    return np.random.choice(['Lose', 'Win', 'Draw'])



# функция, рассчитывающая важность тех или иных параметров для предсказывания
def get_feature_importance(clf, col):
    # важность параметров
    importances = clf.feature_importances_

    fi=pd.DataFrame(col, columns=['feature'])
    fi['importance'] = importances
    fi = fi.sort_values(by='importance', ascending=False)
    fi.index = range(len(fi))
    fi['position'] = fi.index
    return fi



# функция расчета коэффициента в случае угаданного матча
def bets_row(y, venue, predict, bets):
    if y != predict:
        return 0
    else:
        if y == 'Win' and venue == 'Home':
            cf = bets[0]
        elif y == 'Lose' and venue == 'Away':
            cf = bets[0]
        elif y == 'Win' and venue == 'Away':
            cf = bets[2]
        elif y == 'Lose' and venue == 'Home':
            cf = bets[2]
        elif y == 'Draw':
            cf = bets[1]
        return cf * 1

    

# функция расчета метрик
def count_metric(test_df, predict):
    import sklearn.metrics as skm
    # доход
    try:
        tdf = test_df[['y','venue', predict, 'cf1', 'cfdr', 'cf2']]
        test_df['revenue'] = tdf.apply(lambda x: bets_row(x[0], x[1], x[2], [x[3], x[4], x[5]]), axis=1)
        
    except:
        test_df['revenue'] = 0
    
    accurancy = (skm.accuracy_score(test_df['y'], test_df[predict])) * 100
    f1 = (skm.f1_score(test_df['y'], test_df[predict], average='macro')) * 100
    roi =  ((np.sum(test_df['revenue']) / test_df['revenue'].count()) - 1) * 100
    m=[len(test_df),accurancy, f1, roi]
    
    return m



# подсчет метрик по отдельным критериям
def metric_by_segmens(test_df, predict, segment):

    segments=[]
    df_lst=[]
    
    for el in test_df[segment].value_counts().index:
        b=test_df[test_df[segment]==el]
        df_lst.append(b)
        segments.append(el)
    metrics=[]
    for el in df_lst:
        m=count_metric(el, predict)
        metrics.append(m)

    result=pd.DataFrame(metrics)
    result.columns = ['size', 'accuracy', 'f1','roi']
    result['segment'] = segments
    
    return result.sort_values(by=['size', 'f1'], ascending=[False, False])




def model_results(test_df, model_name, predict, predict_parameters):
    
    params = count_metric(test_df, predict)
    params.append(model_name)
    predict_parameters.append(params)
    
    test_df['global_segment'] = 1
    
    model_comparison_df = metric_by_segmens(test_df, predict, 'global_segment')
    model_comparison_df['model'] = model_name
    

    bins = test_df.groupby(['y', predict]).count()[['id']].reset_index()
    bins['y'] = bins['y'].map(score_label_dct)
    bins[predict] = bins[predict].map(score_label_dct)
    bins = bins.sort_values(by=['y', predict])
    bins['total'] = bins.groupby('y').id.transform(np.sum)
    bins['share'] = bins['id'] / bins['total']
    
    # карта для сравнения точности относительно результата матча
    pv = bins.pivot_table(index='y', columns=predict, values='share').fillna(0)

    sns.heatmap(
                pv, 
                cmap="RdYlGn", 
                annot=True,
                fmt=".2f",
                center=0.50,
                )
    
    return model_comparison_df



# функция для разделения датасета на основе временных рамок
def df_date_split(df, train_border1, train_border2, test_border1, test_border2):
    
    train_df = df[(df['date'] >= train_border1) & (df['date'] < train_border2)]
    test_df = df[(df['date'] >= test_border1) & (df['date'] < test_border2)]
    
    test_df = test_df[['id','team', 'opponent', 'venue','team_id', 'opponent_team_id','y']] 

    
    print('train, len:', len(train_df))
    print('test, len:', len(test_df))
    
    return train_df, test_df



# функция для нормализации данных
def normalize_df(df, x_fields_minus_plus, x_fields_zero_one, x_stable):
    
    direct_x = df[x_stable]

    minus_plus_x = pd.DataFrame(
                MinMaxScaler(feature_range=(-1,1)).fit_transform(df[x_fields_minus_plus]), 
                columns=x_fields_minus_plus)

    zero_one_x = pd.DataFrame(
                MinMaxScaler(feature_range=(0,1)).fit_transform(df[x_fields_zero_one]), 
                columns=x_fields_zero_one)

    X = pd.concat([minus_plus_x, zero_one_x, direct_x], axis=1)
    
    return X



# функция для просмотра результата предсказания
def watching(test_df, predict, clf, x_fields, predict_parameters):
    
    leagues = metric_by_segmens(test_df, predict, 'league')
    leagues = leagues.sort_values(by='segment', ascending=False)
    leagues.set_index('segment')[['f1']].plot(kind='barh')
    
    venue = metric_by_segmens(test_df, predict, 'venue')
    
    try:
        bin_size = 100
        test_df['rating_difference_bins'] = np.round(test_df['rating_difference'] / bin_size, 0) * bin_size
        mbs = metric_by_segmens(test_df, predict, 'rating_difference_bins')
        mbs.sort_values(by='segment').set_index('segment')[['f1']].plot()
    except Exception as e:
        print(str(e))
    
    try:
        fi = get_feature_importance(clf, x_fields)
    except:
        fi = pd.DataFrame()
        
    comparison_df = pd.DataFrame(predict_parameters, columns=['games', 'accuracy', 'f1', 'roi', 'model_name']).drop_duplicates()
    comparison_pf = np.round(comparison_df,2)

    return leagues, venue, fi, comparison_df



# функция оценки результата в процентах 
def show_fields(X_train, threshold):
    corr_matrix = []
    col1_lst = []
    col2_lst = []
    for i in X_train.columns:
        for j in X_train.columns:
            r = stats.pearsonr(
                X_train[i], X_train[j]
                                )[0]
            corr_matrix.append(r)
            col1_lst.append(i)
            col2_lst.append(j)
    corr_df = pd.DataFrame(corr_matrix, columns=['correlation'])
    corr_df['field1'] = col1_lst
    corr_df['field2'] = col2_lst

    corr_pv = corr_df.pivot_table(index='field1', columns='field2', values='correlation').fillna(0)#.reset_index()

    problem_df = corr_df[
        (
            (corr_df['correlation'] > threshold) |
            (corr_df['correlation'] < -1 * threshold)
        ) &
        (corr_df['field1'] != corr_df['field2'])
    ]

    sns.heatmap(corr_pv, annot=True, fmt=".2g", cmap="YlGnBu", center=0)
    
    return problem_df




def hypeparams_fit(
                    X_train, y_train, 
                    X_test, test_df,
                    n_range,
                    d_range,
                    ):
    
    finding_lst = []
    for d in d_range:
        print(d)
        for n in n_range:
            print(n)
            clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=22)
            
            clf.fit(X_train, y_train)
            test_df['fitting_hyp_predict']=clf.predict(X_test)
            params = count_metric(test_df, 'fitting_hyp_predict')
            params.append(n)
            params.append(d)
            finding_lst.append(params)
    finding_df = pd.DataFrame(finding_lst)
    finding_df.columns = ['size', 'accuracy', 'f1', 'roi', 'n', 'depth'] 
    
    best_n = finding_df[finding_df['f1']==np.max(finding_df['f1'])]['n'].values[0]
    best_depth = finding_df[finding_df['f1']==np.max(finding_df['f1'])]['depth'].values[0]

    print('best_n', best_n)
    print('best_depth', best_depth)

    for_f1_hm = finding_df.pivot_table(
        index='n', 
        columns='depth', 
        values='f1').fillna(0)
    sns.heatmap(for_f1_hm, annot=True, fmt=".4g", cmap="YlGnBu")
    finding_df[finding_df['depth'] == best_depth].set_index('n')[['f1']].plot()
    finding_df[finding_df['n'] == best_n].set_index('depth')[['f1']].plot()
    
    return best_n, best_depth



# функция прогноза матча не на основе рейтинга Эло    
def no_elo_prob_predict(prob, draw_share):

    try:
        draw_balance=0.5
        if prob>draw_balance+draw_share:
            return 'Lose'
        elif prob<draw_balance-draw_share:
            return 'Win'
        else:
            return 'Draw'
    except:
        return 'Unknown'
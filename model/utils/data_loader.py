from numpy import expand_dims
import pandas as pd

def get(file_path):
    file = pd.read_csv(
        file_path,
        sep='\x01',
        header=None,
        encoding='utf-8',
        names=["features", "labels"]
    )

    data = pd.concat([file["features"].str.split(';', expand=True), pd.DataFrame(file["labels"])], axis=1)
    data.rename(columns={0:'uid', 1:'sid', 2:'tid', 3:'scat', 4:'sage', 5:'slen',
        6:'splay', 7:'tcat', 8:'tage', 9:'tlen', 10:'tplay', 11:'src', 12:'dep',
        13:'time', 'labels':'label'}, inplace=True)
    # data.rename(columns={0:'uid', 1:'sid', 2:'tid', 3:'scat', 4:'sage', 5:'slen',
    #     6:'splay', 7:'tcat', 8:'tage', 9:'tlen', 10:'tplay', 11:'src', 12:'pos', 13:'dep',
    #     14:'time', 'labels':'label'}, inplace=True)
    # data.drop('pos', axis=1, inplace=True)

    # data_pos = data[data['label'] == 1]
    # data_neg = data[data['label'] == 0].sample(n=len(data_pos))
    # data = pd.concat([data_pos, data_neg], ignore_index=True)

    user_set = sorted(data['uid'].unique())
    user_idx = {name : id for id, name in enumerate(user_set)}
    data['uid'] = data['uid'].apply(lambda x : user_idx[x])

    video_set = sorted(data['sid'].append(data['tid'], ignore_index=True).unique())
    video_idx = {name : id for id, name in enumerate(video_set)}
    data['sid'] = data['sid'].apply(lambda x : video_idx[x])
    data['tid'] = data['tid'].apply(lambda x : video_idx[x])

    cat_set = sorted(data['scat'].append(data['tcat'], ignore_index=True).unique())
    cat_idx = {name : id for id, name in enumerate(cat_set)}
    data['scat'] = data['scat'].apply(lambda x : cat_idx[x])
    data['tcat'] = data['tcat'].apply(lambda x : cat_idx[x])

    age_set = sorted(data['sage'].append(data['tage'], ignore_index=True).unique())
    age_idx = {name : id for id, name in enumerate(age_set)}
    data['sage'] = data['sage'].apply(lambda x : age_idx[x])
    data['tage'] = data['tage'].apply(lambda x : age_idx[x])

    len_set = sorted(data['slen'].append(data['tlen'], ignore_index=True).unique())
    len_idx = {name : id for id, name in enumerate(len_set)}
    data['slen'] = data['slen'].apply(lambda x : len_idx[x])
    data['tlen'] = data['tlen'].apply(lambda x : len_idx[x])

    play_set = sorted(data['splay'].append(data['tplay'], ignore_index=True).unique())
    play_idx = {name : id for id, name in enumerate(play_set)}
    data['splay'] = data['splay'].apply(lambda x : play_idx[x])
    data['tplay'] = data['tplay'].apply(lambda x : play_idx[x])

    src_set = sorted(data['src'].unique())
    src_idx = {name : id for id, name in enumerate(src_set)}
    data['src'] = data['src'].apply(lambda x : src_idx[x])

    dep_set = sorted(data['dep'].unique())
    dep_idx = {name : id for id, name in enumerate(dep_set)}
    data['dep'] = data['dep'].apply(lambda x : dep_idx[x])

    set_size = [len(user_set), len(video_set), len(cat_set), len(age_set), len(len_set), len(play_set), len(src_set), len(dep_set)]
    size = data['uid'].size
    train_size = int(size * 0.8)
    
    data = data.sort_values(by='time', ignore_index=True)
    # data = data.sample(frac=1).reset_index(drop=True)
    train_data = data.iloc[0 : train_size]
    test_data = data.iloc[train_size : size]

    x_train = train_data.iloc[:, 0:-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, 0:-1]
    y_test = test_data.iloc[:, -1]

    user_states = pd.DataFrame(expand_dims(data['uid'].drop_duplicates().to_numpy(), axis=1)).sort_values(by=0, ignore_index=True)
    user_states.rename(columns={0:'uid'}, inplace=True)

    video_states_1 = data[['sid', 'scat', 'slen']].drop_duplicates()
    video_states_1.rename(columns={'sid':'vid', 'scat':'cat', 'slen':'len'}, inplace=True)
    video_states_2 = data[['tid', 'tcat', 'tlen']].drop_duplicates()
    video_states_2.rename(columns={'tid':'vid', 'tcat':'cat', 'tlen':'len'}, inplace=True)
    video_states = pd.concat([video_states_1, video_states_2]).drop_duplicates().sort_values(by='vid', ignore_index=True)

    edges = train_data[train_data['label'] == 1]

    return x_train, y_train, x_test, y_test, set_size, user_states, video_states, edges
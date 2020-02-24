from flask import render_template, request
from app import app
import pandas as pd
import requests
# from django.http import HttpResponse
import json
import sqlite3
import os.path
import random
import argparse
import torch
from torch_geometric.datasets import MovieLens
from .pgat_recsys import PGATRecSys
from .utils import get_folder_path

parser = argparse.ArgumentParser()
default_poster_src = 'https://www.nehemiahmfg.com/wp-content/themes/dante/images/default-thumb.png'


########################## Define arguments ##########################
# Dataset params
parser.add_argument("--dataset", type=str, default='movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--step_length", type=int, default=2, help="")
parser.add_argument("--train_ratio", type=float, default=False, help="")
parser.add_argument("--debug", default=0.01, help="")

# Model params
parser.add_argument("--heads", type=int, default=4, help="")
parser.add_argument("--dropout", type=float, default=0.6, help="")
parser.add_argument("--emb_dim", type=int, default=64, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")

# Device params
parser.add_argument("--device", type=str, default='cpu', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")

args = parser.parse_args()

# save id selected by users
iid_list = []
iid_list2 = []
iid_list3 = []
demographic_info = ()

########################## Define arguments ##########################
data_folder, weights_folder, logger_folder = get_folder_path(args.dataset + args.dataset_name)

########################## Setup Device ##########################
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

########################## Define parameters ##########################
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'num_core': args.num_core, 'step_length': args.step_length, 'train_ratio': args.train_ratio,
    'debug': args.debug
}
model_args = {
    'heads': args.heads, 'emb_dim': args.emb_dim,
    'repr_dim': args.repr_dim, 'dropout': args.dropout
}
device_args = {'debug': args.debug, 'device': device, 'gpu_idx': args.gpu_idx}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('device_args params: {}'.format(device_args))

recsys = PGATRecSys(num_recs=10, dataset_args=dataset_args, model_args=model_args, device_args=device_args)
refresh_value = 0

@app.template_global()
def generateIDs(n):
    movie_df = recsys.get_top_n_popular_items(n)
    iids = [iid for iid in movie_df.iid.values]
    return iids

@app.template_global()
def get_movie_name_withID(i):
    i = int(i)
    movies = pd.read_csv('app/ml-1m/movies.dat', sep='::', engine='python')
    movie_name = movies['MovieName'][i]
    return movie_name

@app.template_global()
def get_movie_poster_withID(i):

    # apikey = 'e760129c'
    # apikey = 'e44e5305'
    apikey = '192c6b0e'
    movies = pd.read_csv('app/ml-1m/movies.dat', sep='::', engine='python')
    movie_name = get_movie_name_withID(i)
    movie_title = movie_name[0:-7]
    movie_title = movie_title.replace(' ','+')
    movie_year = movie_name[-5:-1]

    movie_url = "http://www.omdbapi.com/?" + "t=" + movie_title + "&y=" + movie_year + "&apikey=" + apikey
    movie_url_no_year = "http://www.omdbapi.com/?" + "t=" + movie_title + "&apikey=" + apikey
    # print('movie_url: ' + movie_url)

    r = requests.get(movie_url)
    print(r.text)
    response_text = json.loads(r.text)
    try:
        movie_info_dic = response_text
        poster = movie_info_dic['Poster']
        if poster == 'N/A':
            return default_poster_src
        else:
            return poster

    except:
        response_value = response_text['Response']
        # {"Response": "False", "Error": "Movie not found!"}
        if response_value == 'False':
            r2 = requests.get(movie_url_no_year)
            movie_info_dic2 = json.loads(r2.text)
            try:
                poster2 = movie_info_dic2['Poster']
                if poster2 == 'N/A':
                    return default_poster_src
                else:
                    return poster2
            except:
                return default_poster_src


    # except NameError:
    #     r2 = requests.get(movie_url_no_year)
    #     movie_info_dic2 = json.loads(r2.text)
    #     poster2 = movie_info_dic2['Poster']
    #     if poster2 == 'N/A':
    #         return default_poster_src
    #     else:
    #         return poster2
    # except:
    #     return default_poster_src


@app.template_global()
def save_explanation_score_tosqlite(user_id,movie_id,seen_status,explanation,explanation_score,user_study_round):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "MATDemo.db")
    connection = sqlite3.connect(db_path)

    cursor = connection.cursor()
    print("Opened database successfully")

    cursor.execute('create table if not exists EXP_SCORE (user_id,movie_id,seen_status,explanation,explanation_score,user_study_round)')
    params = (user_id,movie_id,seen_status,explanation,explanation_score,user_study_round)

    cursor.execute("INSERT INTO EXP_SCORE VALUES (?,?,?,?,?,?)",params)
    connection.commit()
    print("Records created successfully")

    connection.close()
    return 1


@app.route('/')
@app.route('/index')
def index():
    user = {'username':'Zhe'}
    return render_template('explanation.html', title = 'Film Recommendation', user = user)

@app.route('/explanation')
def explanation():
    return render_template('explanation.html', title = 'Film Recommendation')

@app.route('/user_information')
def user_information():
    return render_template('user_information.html', title = 'Film Recommendation')

@app.route('/user_background')
def user_background():
    return render_template('user_background.html',title = 'Film Recommendation')

@app.route('/movie_preview')
def movie_preview():

    i = range(300)
    movie_ids = generateIDs(300)
    step = 6
    group_movieIDs = [movie_ids[i:i + step] for i in range(0, len(movie_ids), step)]
    click_count = refresh_value
    # import pdb
    # pdb.set_trace()

    return render_template('movie_preview.html',title = 'Film Recommendation', group_movieIDs = group_movieIDs, click_count = click_count)

@app.route('/refresh_count',methods=['GET','POST'])
def refresh_count():
    if request.method == 'POST':
        temp_refresh_value = request.values['refresh_value']
        global refresh_value
        if temp_refresh_value != '':
            refresh_value = int(temp_refresh_value)
            return 'success'

@app.route('/imgID_userinfo_transfer',methods=['GET','POST'])
def imgID_userinfo_transfer():
    global demographic_info
    # import pdb
    # pdb.set_trace()
    if request.method == 'POST':
        the_id = request.values['id']
        the_id = int(the_id)
        gender = request.values['gender']
        occupation = request.values['occupation']

        demographic_info = (gender, occupation)
        iid_list.append(the_id)
        if len(iid_list) == 10:
            print('creating new user...')
        return 'success'
    else:
        return 'fail'

@app.route('/new_iids_for_recommendations',methods=['GET','POST'])
def new_iids_for_recommendations():
    pass


@app.route('/movie_degree')
def movie_degree():
    # import pdb
    # pdb.set_trace()
    global iid_list
    global demographic_info
    recsys.build_user(iid_list, demographic_info)
    print('new user created')

    df, exps = recsys.get_recommendations()
    rec_movie_iids = df.iid.values
    # print(iids)
    # rec_movie_iids = {209,223,234,253,523,1223,334,438,555,619}
    # exps = {'exp209','exp223','exp234','exp253','exp523','exp1223','exp334','exp438','exp555','exp619'}
    return render_template('movie_degree.html',title = 'Film Recommendation',rec_movie_iids_and_explanations = zip(rec_movie_iids,exps))

@app.route('/movie_name_transfer',methods=['GET','POST'])
def movie_name_transfer():
    if request.method == 'POST':
        movie_id = request.values['movie_id']
        movie_name = get_movie_name_withID(movie_id)
        return movie_name

@app.route('/score_movie_transfer',methods=['GET','POST'])
def score_movie_transfer():
    global iid_list2
    if request.method == 'POST':
        user_id = request.values['user_id']
        movie_id = request.values['movie_id']
        seen_status = request.values['seen_status']
        explanation = request.values['explanation']
        score = request.values['score']
        user_study_round = "1"
        print('get new data, user_id:{},movie_id:{},seen_status:{},explanation:{},score:{},user_study_round:{}'.format(user_id,movie_id,seen_status,explanation,score,user_study_round))

        save_explanation_score_tosqlite(user_id,movie_id,seen_status,explanation,score,user_study_round)

        the_id = int(movie_id)
        the_score = int(score)
        if the_score >= 3 :
            # build new iid list with ids which score >= 3
            iid_list2.append(the_id)

        return 'success'
    else:
        return 'fail'

@app.route('/movie_degree2')
def movie_degree2():

    # print(iid_list2)
    global iid_list2
    global demographic_info
    new_iids = recsys.base_iids + iid_list2
    recsys.build_user(new_iids, demographic_info)
    df, exps = recsys.get_recommendations()
    rec_movie_iids2 = df.iid.values
    return render_template('movie_degree2.html',title = 'Film Recommendation',rec_movie_iids_and_explanations2 = zip(rec_movie_iids2,exps))

@app.route('/score_movie_transfer2',methods=['GET','POST'])
def score_movie_transfer2():
    global iid_list3
    if request.method == 'POST':
        user_id = request.values['user_id']
        movie_id = request.values['movie_id']
        seen_status = request.values['seen_status']
        explanation = request.values['explanation']
        score = request.values['score']
        user_study_round = "2"
        print('get new data, user_id:{},movie_id:{},seen_status:{},explanation:{},score:{},user_study_round:{}'.format(user_id,movie_id,seen_status,explanation,score,user_study_round))

        save_explanation_score_tosqlite(user_id,movie_id,seen_status,explanation,score,user_study_round)

        the_id = int(movie_id)
        the_score = int(score)
        if the_score >= 3 :
            # build new iid list with ids which score >= 3
            iid_list3.append(the_id)

        return 'success'
    else:
        return 'fail'

@app.route('/movie_degree3')
def movie_degree3():

    global iid_list3
    global demographic_info
    new_iids = recsys.base_iids + iid_list3
    recsys.build_user(new_iids, demographic_info)
    df, exps = recsys.get_recommendations()
    rec_movie_iids3 = df.iid.values

    return render_template('movie_degree3.html',title = 'Film Recommendation',rec_movie_iids_and_explanations3 = zip(rec_movie_iids3,exps))

@app.route('/score_movie_transfer3',methods=['GET','POST'])
def score_movie_transfer3():
    if request.method == 'POST':
        user_id = request.values['user_id']
        movie_id = request.values['movie_id']
        seen_status = request.values['seen_status']
        explanation = request.values['explanation']
        score = request.values['score']
        user_study_round = "3"
        print('get new data, user_id:{},movie_id:{},seen_status:{},explanation:{},score:{},user_study_round:{}'.format(user_id,movie_id,seen_status,explanation,score,user_study_round))

        save_explanation_score_tosqlite(user_id,movie_id,seen_status,explanation,score,user_study_round)

        return 'success'
    else:
        return 'fail'


@app.route('/user_feedback')
def user_feedback():
    return render_template('user_feedback.html',title = 'Film Recommendation')






from datetime import timedelta
from scipy.sparse import csr_matrix
import polars as pl
import implicit
import numpy as np
import optuna
import mlflow
import argparse


DATA_DIR = 'data\\'
EVAL_DAYS_TRESHOLD = 14


def prepare_data(df_clickstream, df_event):
    treshhold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)
    df_train = df_clickstream.filter(df_clickstream['event_date']<= treshhold)
    df_eval = df_clickstream.filter(df_clickstream['event_date']> treshhold)[['cookie', 'node', 'event']]
    df_eval = df_eval.join(df_train, on=['cookie', 'node'], how='anti')
    df_eval = df_eval.filter(
    pl.col('event').is_in(
        df_event.filter(pl.col('is_contact')==1)['event'].unique()
    )
    )
    df_eval = df_eval.filter(
        pl.col('cookie').is_in(df_train['cookie'].unique())
    ).filter(
        pl.col('node').is_in(df_train['node'].unique())
    )
    df_eval = df_eval.unique(['cookie', 'node'])
    return df_train, df_eval


def recall_at(df_true, df_pred, k=40):
    return  df_true[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']], 
        how='left',
        on = ['cookie', 'node']
    ).select(
        [pl.col('value').fill_null(0), 'cookie']
    ).group_by(
        'cookie'
    ).agg(
        [
            pl.col('value').sum()/pl.col(
                'value'
            ).count()
        ]
    )['value'].mean()


def get_pred(users, nodes, user_to_pred, params):
    user_ids = users.unique().to_list()
    item_ids = nodes.unique().to_list()
        
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_item_id = {v:k for k,v in item_id_to_index.items()}
    
    rows = users.replace_strict(user_id_to_index).to_list()
    cols = nodes.replace_strict(item_id_to_index).to_list()
    
    values = [1] * len(users)
    
    sparse_matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    model = implicit.als.AlternatingLeastSquares(**params)     
    model.fit(sparse_matrix, )    
    
    user4pred = np.array([user_id_to_index[i] for i in user_to_pred])
    
    recommendations, scores = model.recommend(user4pred, sparse_matrix[user4pred], N=40, filter_already_liked_items=True)
    
    df_pred = pl.DataFrame(
        {
            'node': [
                [index_to_item_id[i] for i in i] for i in recommendations.tolist()
            ], 
             'cookie': list(user_to_pred),
            'scores': scores.tolist()
            
        }
    )
    df_pred = df_pred.explode(['node', 'scores'])
    return df_pred


def objective_als(df_train, df_eval, params, run_name, experiment_id):    
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        mlflow.log_params(params)

        eval_users = df_eval['cookie'].unique().to_list()
        df_pred = get_pred(df_train["cookie"], df_train["node"], eval_users, params)
        recall = recall_at(df_eval, df_pred, k=40)

        mlflow.log_metric('Recall_40', recall)
                
    return recall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--factors', type=int, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--regularization', type=float, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    df_clickstream = pl.read_parquet(f'{DATA_DIR}clickstream.pq')
    df_event = pl.read_parquet(f'{DATA_DIR}events.pq')

    df_train, df_eval = prepare_data(df_clickstream, df_event)

    mlflow.set_tracking_uri('http://51.250.35.156:5000/')
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        experiment_id = mlflow.create_experiment(args.experiment)
    else:
        experiment_id = experiment.experiment_id

    params = {'factors': args.factors, 'iterations': args.iterations, 'alpha': args.alpha, 'regularization': args.regularization}
    recall = objective_als(df_train, df_eval, params, args.run_name, experiment_id)
    print(recall)
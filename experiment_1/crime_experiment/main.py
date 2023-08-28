import pandas as pd
import argparse
import os
import pickle

from experiment import run_exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment file for fairness testing')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Required argument: relative path of the dataset to process')
    args = parser.parse_args()
    
    data = pd.read_csv(args.dataset
    )
    model, report = run_exp(data)
    os.makedirs('ris', exist_ok=True)
    dir_name = os.path.basename(os.path.dirname(__file__))
    report.round(3).to_csv(os.path.join('ris',f'report_{dir_name}.csv'))
    model_name = f"{report.loc[0,'model']}_{report.loc[0,'fairness_method']}"
    pickle.dump(model, open(os.path.join('ris',model_name+'_'+dir_name+'.pkl'), 'wb'))
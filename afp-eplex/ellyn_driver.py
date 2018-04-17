mport numpy as np
import pandas as pd
from ellyn import ellyn
from sklearn.model_selection import train_test_split
from sklearn.metrics import SCORERS, r2_score, mean_absolute_error
from time import time
from sklearn.preprocessing import StandardScaler

TARGET_NAME = 'target'
INPUT_SEPARATOR = '\t'
n_jobs = 10

def main(inputfile, random_state):
    input_data = pd.read_csv(
            inputfile,
            sep=INPUT_SEPARATOR,
            dtype=np.float64,
    )

    if TARGET_NAME not in input_data.columns.values:
        raise ValueError(
            'The provided data file does not seem to have a target column. '
            'Please make sure to specify the target column using the -target '
            'parameter.'
        )

    # data
    sc_y = StandardScaler()
    X = StandardScaler().fit_transform(input_data.drop(TARGET_NAME, axis=1).values.astype(float))
    y = sc_y.fit_transform(input_data[TARGET_NAME].values.reshape(-1,1))

    # operator set
    ops = 'n,v,+,-,*,/,sin,cos,exp,log,2,3'
    ops_w = '6,6,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5'


#    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=None)

    # Create the pipeline for the model
    est = ellyn(g=10,popsize=10,selection='epsilon_lexicase',
                lex_eps_global=False,
                lex_eps_dynamic=False,
                fit_type='MSE',max_len=100,
                islands=True,
                num_islands=n_jobs,
                island_gens=100,
                verbosity=0,
                print_data=False,
                elitism=True,
                ops=ops,
                ops_w=ops_w, pHC_on=True,prto_arch_on=True, random_state=1)

    #fit model
    # pdb.set_trace()
    t0 = time()
    est.fit(X_train,y_train)
    #get fit time
    runtime = time()-t0

    training_score_mae = mean_absolute_error(sc_y.inverse_transform(est.predict(X_train)),
                                  sc_y.inverse_transform(y_train))
    training_score_r2 = r2_score(sc_y.inverse_transform(est.predict(X_train)),
                                sc_y.inverse_transform(y_train))

    holdout_score_mae = mean_absolute_error(sc_y.inverse_transform(est.predict(X_test)),
                                  sc_y.inverse_transform(y_test))
    holdout_score_r2 = r2_score(sc_y.inverse_transform(est.predict(X_test)),
                                sc_y.inverse_transform(y_test))


    print('\nTraining score (MAE): {}'.format(training_score_mae))
    print('Training score (r2): {}'.format(training_score_r2))
    print('Holdout score (MAE): {}'.format(holdout_score_mae))
    print('Holdout score (r2): {}'.format(holdout_score_r2))
    print('Fitting Time: {}'.format(runtime))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Perform Ellyn")
    parser.add_argument('--inputfile', dest='inputfile', default='/home/weixuanf/AI/penn-ml-benchmarks/datasets/regression/663_rabe_266/663_rabe_266.tsv.gz')
    parser.add_argument('--random_state', dest='random_state', default=42)

    params = vars(parser.parse_args())
    inputfile = str(params['inputfile'])
    random_state = int(params['random_state'])

    print('INPUT_FILE\t=\t{}'.format(inputfile))
    print('RANDOM_STATE\t=\t{}'.format(random_state))

    main(inputfile, random_state)


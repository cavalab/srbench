from ellyn import ellyn

hyper_params = [
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (64,),
        'rt_cross':(0.2,),'rt_mut':(0.8,),
    },
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (64,),
        'rt_cross':(0.8,),'rt_mut':(0.2,),
    },
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (64,),
        'rt_cross':(0.5,),'rt_mut':(0.5,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (64,),
        'rt_cross':(0.2,),'rt_mut':(0.8,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (64,),
        'rt_cross':(0.8,),'rt_mut':(0.2,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (64,),
        'rt_cross':(0.5,),'rt_mut':(0.5,),
    },

]


# Create the pipeline for the model
est = ellyn(selection='eplex',
            lex_eps_global=False,
            lex_eps_dynamic=False,
            islands=True,
            num_islands=1,
            island_gens=1,
            verbosity=0,
            print_data=False,
            elitism=True,
            pHC_on=True,
            prto_arch_on=True)


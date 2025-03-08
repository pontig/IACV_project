def objective_function(Ps, alphas, betas, splines, tss):
    score = 0
    for i in range(len(Ps)):
        for j in range
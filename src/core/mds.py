import numpy
import scipy

def simulated_annealing_mds(
            points, dim=2,
            start_grid_size=10,
            stop_grid_size=1000,
            startT=500.0,
            stopT = 10,
            threshold_factor = 0.1,
            max_iterations_without_improvement=1000
):

    # find variables with maximum variance
    variance = numpy.var(points, 0)
    sorting = numpy.argsort(variance)
    X = points[:,sorting[-dim:]]
    best_X = X.copy()

    # calculate distances in multidimensional space
    delta = scipy.spatial.distance.pdist(points)
    delta_m = scipy.spatial.distance.squareform(delta)
    # calculate distances in reduced-dimension space
    dist = scipy.spatial.distance.pdist(X)
    # calculate fitness of the mapping
    start_fitness = numpy.sum( (delta-dist)**2 )
    fitness = start_fitness
    best_fitness = fitness

    fitness_threshold = threshold_factor * fitness

    print 'starting with fitness=%f' % fitness

    T = startT
    dT = 0.0

    #for n in xrange(iterations):
    n = 0
    m = 0
    last_grid_size_with_improvement = start_grid_size
    while fitness > fitness_threshold:
        
        if m > max_iterations_without_improvement:
            break
        
        # update grid_size
        grid_size = last_grid_size_with_improvement  + (stop_grid_size - last_grid_size_with_improvement ) \
                * m / float(max_iterations_without_improvement)
        #grid_size = stop_grid_size \
        #    + ( start_grid_size - stop_grid_size ) \
        #        * numpy.exp( - ( start_fitness - fitness ) / float( fitness) )
        # set step-size for choosing a neighbour        
        step_size = (numpy.max(points,0)-numpy.min(points,0)) / grid_size

        # set temperature
        #T = startT * ( (fitness - fitness_threshold) / (start_fitness - fitness_threshold) )
        T = stopT \
            + ( startT-stopT ) \
                * numpy.exp( - ( start_fitness - fitness ) / float( fitness) )
#            if T >= 0.5*startT:
#                T -= startT / iterations
#            else:
#                if dT == 0.0:
#                    approxIterations = 2 * fitness * n / float(start_fitness - fitness)
#                    dT = T / approxIterations
#                    if dT < 0:
#                        break
#                T -= dT
        n += 1
        m += 1
        if (n % 1000) == 0:
            print 'fitness=%f, T=%f, n=%d, m=%d, grid=%f, step1=%f, step2=%f' % (fitness,T,n,m,grid_size,step_size[0],step_size[1])
        
        # choose new neighbour
        l = numpy.random.randint(0, points.shape[0])
        j = numpy.random.randint(0, dim)
        b = numpy.random.randint(0, 2)
        b = 1 - 2*b
        new_X_row = numpy.array(X[l])
        new_X_row[j] += b * step_size[j]
        
        # calculate changes in distances
        new_dist_to_point = scipy.spatial.distance.cdist( (new_X_row,) , X )
        old_dist_to_point = scipy.spatial.distance.cdist( (X[l],) , X )
        
        # calculate new fitness
        dfitness1 = numpy.sum( (delta_m[l] - new_dist_to_point)**2 )
        dfitness2 = numpy.sum( (delta_m[l] - old_dist_to_point)**2 )
        new_fitness = fitness + dfitness1 - dfitness2
        
        # found a better solution?
        if new_fitness <= fitness:
            # yep, keep it
            fitness = new_fitness
            X[l] = new_X_row
            # and see if it's the best solution we had so far
            if fitness <= best_fitness:
                m = 0
                #last_grid_size_with_improvement = 
                best_X = X.copy()
                best_fitness = fitness
        else:
            # the new solution is worse. but do we still keep it?
            p = numpy.exp( - (new_fitness - fitness) / T)
            if p > numpy.random.random():
                fitness = new_fitness
                X[l] = new_X_row

    if fitness <= fitness_threshold:
        print 'reached threshold'
    else:
        print 'timeout reached'
        
    return best_X,best_fitness

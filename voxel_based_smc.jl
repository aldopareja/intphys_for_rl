using Gen, Seaborn

@dist num_objects_dist(p) = neg_binom(3, p) + 1

@gen function multi_object(T::Int, num_objects::Int)

    measurement_noise = 0.005
    velocity_var = 1e-5

    num_dims = 2
    #prior on the number of objects in a scene
    # num_objects ~ num_objects_dist(0.9)

    xs = Array{Float64}(undef, num_objects, T, 2)

    #prior on velocity
    v = [{(:v,o,1,i)} ~ uniform(-0.015,0.015) 
         for o=1:num_objects ,i=1:num_dims]
    
    #prior on position
    x = [{(:x,o,1,i)} ~ uniform(0,1) 
         for o=1:num_objects ,i=1:num_dims]
    
    #initial positions
    [{(:z,o,1,i)} ~ normal(x[o,i], measurement_noise) 
     for o=1:num_objects, i=1:num_dims]

    for t in 2:T
        x += v

        v = [{(:v,o,t,i)} ~ normal(v[o,i],velocity_var) 
             for o=1:num_objects, i=1:num_dims]

        [{(:z,o,t,i)} ~ normal(x[o,i], measurement_noise) 
         for o=1:num_objects, i=1:num_dims]
    end
end

trace = Gen.simulate(test, (10,10))
Gen.get_choices(trace)

sims = [Gen.simulate(test, ())[:num_objects] for i in 1:1000]
displot(sims)
display(gcf())
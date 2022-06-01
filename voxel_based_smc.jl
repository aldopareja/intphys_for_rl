using Gen
includet("viz.jl")

"""
returns a function that computes what slots in a unit 2d grid are occupied given a set of object
positions.
"""
@gen function get_occupancy_grid(object_poses::Array{Float64,2},
    side_num_slots::Int,
    false_positive_noise::Real,
    false_negative_noise::Real)
    bin_size = 1 / side_num_slots
    idx = convert.(Int, (object_poses .÷ bin_size)) .+ 1
    sparse_occupancy_grid = [tuple(idx[o, :]...) for o = 1:size(idx, 1)]

    occupancy_grid = Array{Bool}(undef, side_num_slots, side_num_slots)
    for i = 1:side_num_slots, j = 1:side_num_slots
        if (i, j) in sparse_occupancy_grid
            occupancy_grid[i, j] = {(:z, i, j)} ~ bernoulli(1 - false_negative_noise)
        else
            occupancy_grid[i, j] = {(:z, i, j)} ~ bernoulli(false_positive_noise)
        end
    end
    return occupancy_grid
end


"""
returns an 2d occupancy grid over time depending on the movement of latent object trajectories.
"""
@gen function multi_object(T::Int, num_objects::Int, side_num_slots::Int,
                            false_postive_noise::Real, false_negative_noise::Real,
                            velocity_var::Real, max_init_vel::Real)


    measurement_noise = 0.005
    num_dims = 2

    #prior on the number of objects in a scene
    # num_objects ~ num_objects_dist(0.9)

    xs = Array{Float64}(undef, T, num_objects, num_dims)
    occupancy_grid_time = Array{Bool}(undef, T, side_num_slots, side_num_slots)

    #prior on velocity
    v = [{(:v, 1, o, i)} ~ uniform(-max_init_vel, max_init_vel)
         for o = 1:num_objects, i = 1:num_dims]

    #prior on position
    x = [{(:x, 1, o, i)} ~ normal(0.5, 0.2)
        #  {(:x, 1, o, i)} ~ uniform(0, 1)
         for o = 1:num_objects, i = 1:num_dims]

    xs[1, :, :] = x

    #initial measurements
    occupancy_grid_time[1, :, :] = {(:occupancy_grid, 1)} ~ get_occupancy_grid(x,
        side_num_slots,
        false_postive_noise,
        false_negative_noise)

    for t in 2:T
        x += v
        xs[t, :, :] = x

        v = [{(:v, t, o, i)} ~ normal(v[o, i], velocity_var)
             for o = 1:num_objects, i = 1:num_dims]

        occupancy_grid_time[t, :, :] = {(:occupancy_grid, t)} ~ get_occupancy_grid(x,
            side_num_slots,
            false_postive_noise,
            false_negative_noise)
    end
    return xs, occupancy_grid_time
end


function choicemap_from_grid(t::Int, occupancy_grid_time::Array{Bool,3})

    num_bins = size(occupancy_grid_time)[end]
    #choices of the form:((:occupancy_grid,t)=>(:z, i, j),Bool)
    choices = Vector{Tuple{Pair{Tuple{Symbol,Int64},Tuple{Symbol,Int64,Int64}},Bool}}(undef, num_bins^2)

    for k in 1:num_bins^2
        i = (k - 1) ÷ num_bins + 1
        j = (k - 1) % num_bins + 1
        choices[k] = ((:occupancy_grid, t) => (:z, i, j),
            occupancy_grid_time[t, i, j])
    end
    return Gen.choicemap(choices...)
end

function particle_filter(num_particles::Int, occupancy_grid_time::Array{Bool,3}, num_samples::Int,
    model_args::Tuple)

    init_obs = choicemap_from_grid(1, occupancy_grid_time)
    static_args = model_args[2:end]
    init_args = (1, static_args...) #change the time

    state = Gen.initialize_particle_filter(multi_object, init_args, init_obs, num_particles)

    T = size(occupancy_grid_time, 1)
    num_objects = model_args[2]

    for t = 2:T
        # apply a rejuvenation move to each particle only on one object/dim at a time
        for i=1:num_particles
            initial_choices = Array{Tuple{Symbol, Int64, Int64, Int64}}(undef, 2)
            o = i % num_objects + 1
            d = i % 2 + 1
            initial_choices[1] = (:x,1,o,d)
            initial_choices[2] = (:v,1,o,d)
            initial_choices = select(initial_choices...)
            state.traces[i], _  = mh(state.traces[i], initial_choices)
        end

        Gen.maybe_resample!(state, ess_threshold=num_particles / 2)
        obs = choicemap_from_grid(t, occupancy_grid_time)
        args = tuple(t, static_args...)
        Gen.particle_filter_step!(state, args, (UnknownChange(), [NoChange() for _ in 1:length(static_args)]...),
                                  obs)
    end

    return Gen.sample_unweighted_traces(state, num_samples)
end


trace = Gen.simulate(multi_object, (100, 5, 20, 0.0002, 0.001, 5e-2, 0.05))
xs, occupancy_grid_time = Gen.get_retval(trace)
visualize() do 
    draw_observation(occupancy_grid_time)
    draw_object_trajectories_different_colors(xs)
end
@time pf_traces = particle_filter(100, occupancy_grid_time, 200, Gen.get_args(trace));
visualize() do 
    xs, occupancy_grid_time = Gen.get_retval(trace)
    draw_observation(occupancy_grid_time)
    draw_object_trajectories_single_color(xs)
    for tr in pf_traces
        xs = Gen.get_retval(trace)
        draw_object_trajectories_single_color(xs; color="red", overlay="true")
    end
end

vizualize() do 
    render_trace(trace3) 
    render_trace(trace2)
    background("blanchedalmond")
end


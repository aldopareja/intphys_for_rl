using Gen, Seaborn

"""
returns a function that computes what slots in a unit 2d grid are occupied given a set of object
positions.
"""
@gen function get_occupancy_grid(object_poses::Array{Float64,2},
                                side_num_slots::Int,
                                false_postive_noise::Real,
                                false_negative_noise::Real)
    bin_size = 1 / side_num_slots
    sparse_occupancy_grid = Vector{Tuple{Int64,Int64}}()
    for i = 1:side_num_slots, j = 1:side_num_slots
        if ({(:false_pos, i, j)} ~ bernoulli(false_postive_noise))
            sparse_occupancy_grid = [sparse_occupancy_grid; (i, j)]
        end
    end
    num_objects = size(object_poses)[1]
    for o in 1:num_objects
        pos = object_poses[o, :]

        #ignore out of viewfield objects
        if ~(all(0.0 .<= pos .<= 1.0)) ||
           ({(:false_neg, o)} ~ bernoulli(false_negative_noise))
            continue
        end
        idx1, idx2 = convert.(Int, (pos .÷ bin_size)) .+ 1
        sparse_occupancy_grid = [sparse_occupancy_grid; (idx1, idx2)]
    end
    return sparse_occupancy_grid
end

# @dist num_objects_dist(p) = neg_binom(3, p) + 1


"""
returns an 2d occupancy grid over time depending on the movement of latent object trajectories.
"""
@gen function multi_object(T::Int, num_objects::Int, side_num_slots::Int,
                            false_postive_noise::Real, false_negative_noise::Real,
                            velocity_var)

    measurement_noise = 0.005
    # velocity_var = 3e-3

    num_dims = 2
    #prior on the number of objects in a scene
    # num_objects ~ num_objects_dist(0.9)

    occupancy_grid_time = Vector{Vector{Tuple{Int64, Int64}}}(undef, T)

    #prior on velocity
    v = [{(:v, 1, o, i)} ~ uniform(-0.015, 0.015)
         for o = 1:num_objects, i = 1:num_dims]

    #prior on position
    x = [{(:x, 1, o, i)} ~ uniform(0, 1)
         for o = 1:num_objects, i = 1:num_dims]

    #initial measurements
    occupancy_grid_time[1] = {(:occupancy_grid, 1)} ~ get_occupancy_grid(x,
        side_num_slots,
        false_postive_noise,
        false_negative_noise)

    for t in 2:T
        x += v

        v = [{(:v, t, o, i)} ~ normal(v[o, i], velocity_var)
             for o = 1:num_objects, i = 1:num_dims]

        occupancy_grid_time[t] = {(:occupancy_grid, t)} ~ get_occupancy_grid(x,
            side_num_slots,
            false_postive_noise,
            false_negative_noise)
    end
    return occupancy_grid_time
end

function render_trace(trace)
    (T, num_objects, side_num_slots,
    _, _) = Gen.get_args(trace)

    occupancy_grid_time = Gen.get_retval(trace)
    
    Drawing(1000,1000)
    background("cyan")
    setcolor("red")
    origin()
    cell_size = 1000÷side_num_slots
    cells = Table(side_num_slots,side_num_slots, cell_size, cell_size)
    for t in 1:T
        for (r,c) in occupancy_grid_time[t]
            box(cells[r,c], cell_size,cell_size, :fill)
        end
    end
    finish()
    preview()
end

trace = Gen.simulate(multi_object, (100, 5, 50, 0.000005, 0.00001))
render_trace(trace)
using Gen
import LinearAlgebra
includet("viz.jl")

DEBUG = true

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
# @gen function multi_object(T::Int, num_objects::Int, side_num_slots::Int,
#   false_postive_noise::Real, false_negative_noise::Real,
#   velocity_var::Real, max_init_vel::Real)


#   measurement_noise = 0.005
#   num_dims = 2

#   #prior on the number of objects in a scene
#   # num_objects ~ num_objects_dist(0.9)

#   xs = Array{Float64}(undef, T, num_objects, num_dims)
#   occupancy_grid_time = Array{Bool}(undef, T, side_num_slots, side_num_slots)



#   #prior on position
#   x = [
#     # {(:x, 1, o, i)} ~ normal(0.5, 0.2)
#     {(:x, 1, o, i)} ~ uniform(0, 1)
#     for o = 1:num_objects, i = 1:num_dims]

#   xs[1, :, :] = x

#   #initial measurements
#   occupancy_grid_time[1, :, :] = {(:occupancy_grid, 1)} ~ get_occupancy_grid(x,
#     side_num_slots,
#     false_postive_noise,
#     false_negative_noise)

#   for t in 2:T
#     v = [{(:v, t, o, i)} ~ normal(v[o, i], velocity_var)
#          for o = 1:num_objects, i = 1:num_dims]

#     x += v
#     xs[t, :, :] = x

#     occupancy_grid_time[t, :, :] = {(:occupancy_grid, t)} ~ get_occupancy_grid(x,
#       side_num_slots,
#       false_postive_noise,
#       false_negative_noise)
#   end
#   return xs, occupancy_grid_time
# end

@gen function multi_object(T::Int, num_objects::Int, side_num_slots::Int,
  false_postive_noise::Real, false_negative_noise::Real,
  velocity_var::Real, max_init_vel::Real)


  # measurement_noise = 0.005
  num_dims = 2

  #prior on the number of objects in a scene
  # num_objects ~ num_objects_dist(0.9)

  xs = Array{Float64}(undef, T+1, num_objects, num_dims)
  occupancy_grid_time = Array{Bool}(undef, T+1, side_num_slots, side_num_slots)

  #prior on position
  x = [
    # {(:x, 1, o, i)} ~ normal(0.5, 0.2)
    {(:x, 1, o, i)} ~ uniform(0, 1)
    for o = 1:num_objects, i = 1:num_dims]

  xs[1, :, :] = x

  #initial measurements
  occupancy_grid_time[1, :, :] = {(:occupancy_grid, 1)} ~ get_occupancy_grid(x,
                                side_num_slots,
                                false_postive_noise,
                                false_negative_noise)
  local v
  for t in 1:T
    if t==1
      #prior on velocity
      v = [{(:v, t, o, i)} ~ uniform(-max_init_vel, max_init_vel)
           for o = 1:num_objects, i = 1:num_dims]
      
    else
      v = [{(:v, t, o, i)} ~ normal(v[o, i], velocity_var)
         for o = 1:num_objects, i = 1:num_dims]
    end
    x += v

    xs[t+1, :, :] = x
    occupancy_grid_time[t+1, :, :] = {(:occupancy_grid, t+1)} ~ get_occupancy_grid(x,
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

  static_args = model_args[2:end]
  init_args = (0, static_args...) #change the time to 1 since only the first observation is in the first part
  T = model_args[1]
  num_objects = model_args[2]
  num_side_slots = model_args[3]
  high_score_proposal = float(num_side_slots^2)
  init_proposal_args = (num_objects, occupancy_grid_time, 1, high_score_proposal)

  init_obs = choicemap_from_grid(1, occupancy_grid_time)
  state = Gen.initialize_particle_filter(multi_object, init_args, init_obs,
    initial_pos_proposal_pf, init_proposal_args,
    num_particles)
  
  # return state.traces

  # for _=1:20
  #   [state.traces[i] = init_pos_perturbation_move(state.traces[i],num_objects, high_score_proposal,
  #                                                 occupancy_grid_time)
  #    for i in 1:num_particles]
  # end

  # mh_choices = [(:v, 1, o, d) for o = 1:num_objects, d = 1:2][:]

  for t = 1:T
    # apply a rejuvenation move to each particle only on one object/vel_dim at a time
    accepts = Vector{Bool}(undef, num_particles*num_objects)
    for i=1:num_particles
      state.traces[i], a = Gen.mh(state.traces[i], initial_pos_proposal_mh, (num_objects, occupancy_grid_time, 1, high_score_proposal))
      accepts[i] = a
    end
    @show sum(accepts), length(accepts)

    # mh_choices = [mh_choices; [(:v, t, o, d) for o = 1:num_objects, d = 1:2][:]]
    Gen.maybe_resample!(state, ess_threshold=num_particles / 8)
    obs = choicemap_from_grid(t+1, occupancy_grid_time)
    args = tuple(t, static_args...)
    Gen.particle_filter_step!(state, args, (UnknownChange(), [NoChange() for _ in 1:length(static_args)]...),
      obs)

    accepts = Vector{Bool}(undef, num_particles*num_objects)
    for i=1:num_particles
      state.traces[i], a = Gen.mh(state.traces[i], vel_t_proposal, (t, 1, occupancy_grid_time, high_score_proposal))
      accepts[i] = a
    end
    @show "vel", sum(accepts), length(accepts)
    # accepts = Vector{Bool}(undef, num_particles*num_objects)
    # for i=1:num_particles
    #   for o in 1:num_objects
    #     state.traces[i],a = Gen.mh(state.traces[i], vel_t_proposal, (t, o, occupancy_grid_time, high_score_proposal))
    #     accepts[(i-1)* num_objects + o] = a
    #   end
    # end
    # @show sum(accepts), length(accepts)
  end

  return Gen.sample_unweighted_traces(state, num_samples)
end


"""
samples a 2d piecewise uniform over the unit square given a matrix of scores, each element in scores is proportional
to the probability of a point in the bin represented by it.
"""
@gen function piecewise_uniform_2d_unit(scores::Array{Float64,2},
  addrCol::Tuple{Symbol,Int64,Int64,Int64},
  addrRow::Tuple{Symbol,Int64,Int64,Int64},
  anchorRow::Float64,
  anchorCol::Float64)

  num_bins = size(scores)[1]
  bin_size = 1 / num_bins
  bounds = collect(0.0:bin_size:1.0)
  
  boundsCol = bounds .- anchorCol
  probs = sum(scores; dims=1)[:] # compute the marginal
  probs /= sum(probs)
  col = {addrCol} ~ piecewise_uniform(boundsCol, probs)
  
  boundsRow = bounds .- anchorRow
  col_bin_idx = trunc(Int, (col + anchorCol) ÷ bin_size) + 1 #find the sampled column

  if col_bin_idx > num_bins
    DEBUG && @show col_bin_idx, col, anchorCol, bin_size
    col_bin_idx = num_bins
  end
  probs = scores[:, col_bin_idx]
  probs /= sum(probs)
  row = {addrRow} ~ piecewise_uniform(boundsRow, probs)
  return row, col
end

# @gen function init_pos_proposal_mh(prev_trace, scores::Array{Float64,2},
#                     addrCol::Tuple{Symbol,Int64,Int64,Int64},
#                     addrRow::Tuple{Symbol,Int64,Int64,Int64},
#                     anchorX::Float64,
#                     anchorY::Float64)
#   @trace(piecewise_uniform_2d_unit(scores,addrCol,addrRow,anchorX,anchorY))
# end

@gen function vel_t_proposal(trace, t, obj_id, occupancy_grid_time, high_score)
  xs, _ = Gen.get_retval(trace)
  row,col = xs[t,obj_id,:]
  scores = scores_from_occupancy(occupancy_grid_time, (t+1, t+1), high_score)
  vrow, vcol = piecewise_uniform_2d_unit(scores, (:na,1,1,1), (:na,1,1,2), row,col)
  #TODO: change the standard deviation
  vrow = {(:v,t,obj_id,1)} ~ normal(vrow, 1e-3)
  vcol = {(:v,t,obj_id,2)} ~ normal(vcol, 1e-3)
end

# function vel_t_perturb(trace, t)

# end

function scores_from_occupancy(occupancy_grid_time::Array{Bool,3}, interval::Tuple{Int,Int},
                               high_score::Float64)
  #reduce occupancy grid to a single matrix
  scores = occupancy_grid_time[(:)(interval...), :, :]
  scores = reduce((x,y) -> x.|y, scores; 
                              dims=1, init=false)[1,:,:]
  
  #assign a score to occupied grids
  scores = scores .* (high_score - 1) .+ 1.0
  return scores
end

@gen function initial_pos_proposal_mh(trace,num_objects::Int, occupancy_grid_time::Array{Bool,3},
  num_init_time_steps::Int, high_score::Float64)

  scores = scores_from_occupancy(occupancy_grid_time, (1,num_init_time_steps), high_score)

  for o=1:num_objects
    @trace(piecewise_uniform_2d_unit(scores, (:x,1,o,2), (:x,1,o,1), 0.0, 0.0))
  end
end

@gen function initial_pos_proposal_pf(num_objects::Int, occupancy_grid_time::Array{Bool,3},
  num_init_time_steps::Int, high_score::Float64)

  scores = scores_from_occupancy(occupancy_grid_time, (1,num_init_time_steps), high_score)

  for o=1:num_objects
    @trace(piecewise_uniform_2d_unit(scores, (:x,1,o,2), (:x,1,o,1), 0.0, 0.0))
  end
end

"""
perturb the initial position with a proposal giving high probability to occupied bins in time 1.
then it perturbs the initial velocity such that the implied second position lies (with high probability) in
an occupied bin on the second time step
"""
function init_pos_perturbation_move(trace, num_objects, high_score, occupancy_grid_time)
    scores1 = scores_from_occupancy(occupancy_grid_time, (1,1), high_score)
    scores2 = scores_from_occupancy(occupancy_grid_time, (2,2), high_score)
    for o=1:num_objects
        trace, _ = Gen.mh(trace, init_pos_proposal_mh, (scores1, (:x,1,o,2), (:x,1,o,1), 0.0, 0.0),)
        # anchorX, anchorY = [Gen.get_choice(trace, (:x,1,o,i)).retval for i=1:2]
        # trace, _ = Gen.mh(trace, init_pos_proposal_mh, (scores2, (:v,1,o,2), (:v,1,o,1), anchorX, anchorY))
    end
    return trace
end


trace = Gen.simulate(multi_object, (10, 1, 20, 0.0002, 0.001, 2e-2, 0.05))
xs, occupancy_grid_time = Gen.get_retval(trace)
visualize() do
  draw_observation(occupancy_grid_time[:,:,:])
  draw_object_trajectories_different_colors(xs)
end
@time pf_traces = particle_filter(1000, occupancy_grid_time, 400, Gen.get_args(trace));
visualize() do
  xs1, occupancy_grid_time = Gen.get_retval(trace)
  draw_observation(occupancy_grid_time)
  for tr in pf_traces
    xs, _ = Gen.get_retval(tr)
    draw_object_trajectories_single_color(xs; color="red", overlay=true)
  end
  draw_object_trajectories_single_color(xs1)
end
export DiffEqObjective, build_loss_objective

struct DiffEqObjective{F,G} <: Function
  cost_function::F #objective function
  cost_gradient!::G #gradient update function
end

# build function-like objects, if take 1 parameter, it's an objective function
# if take 2 parameters, it's a gradient update function
(f::DiffEqObjective)(x) = f.cost_function(x)
(f::DiffEqObjective)(result,x) = f.cost_gradient!(result,x)

function loss_split{T<:Vector, W<:Union{Vector, Void}}(loss::L2Loss{T, T, W}, shooting)
  #missing values error
  @assert length(loss.t) == length(loss.data) "time and data length mismatched"
  len = length(loss.t)
  # get subinterval length
  sub_len = div(len, shooting)
  # mass shooting error
  @assert sub_len > 0 "shooting number lager than data length"
  #split loss instance into subintervals, return an array of loss subintervals
  if loss.weight == nothing
    [L2Loss(loss.t[i:i+sub_len], loss.data[i:i+sub_len], nothing) for i in 1:sub_len:shooting]
  else
    [L2Loss(loss.t[i:i+sub_len], loss.data[i:i+sub_len], loss.weight[i:i+sub_len]) for i in 1:sub_len:shooting]
  end
end

function loss_split{T<:Vector, M<:Matrix, W<:Union{Vector, Void}}(loss::L2Loss{T, M, W}, shooting)
  #missing values error
  @assert length(loss.t) == size(loss.data)[2] "time and data length mismatched"
  len = length(loss.t)
  # get subinterval length
  sub_len = div(len, shooting)
  # mass shooting error
  @assert sub_len > 0 "shooting number lager than data length"
  #split loss instance into subintervals, return an array of loss subintervals
  if loss.weight == nothing
    [L2Loss(loss.t[i:i+sub_len], loss.data[:,i:i+sub_len], nothing) for i in 1:sub_len:shooting]
  else
    [L2Loss(loss.t[i:i+sub_len], loss.data[i:i+sub_len], loss.weight[i:i+sub_len]) for i in 1:sub_len:shooting]
  end
end

function build_loss_objective(prob::DEProblem,alg,loss,regularization=nothing;
                              mpg_autodiff = false,
                              shooting::Int = 1,
                              verbose_opt = false,
                              verbose_steps = 100,
                              prob_generator = problem_new_parameters,
                              autodiff_prototype = mpg_autodiff ? zeros(prob.p) : nothing,
                              autodiff_chunk = mpg_autodiff ? ForwardDiff.Chunk(autodiff_prototype) : nothing,
                              kwargs...)
  # keep track of # function evaluations
  count = 0
  function verbose(count,verbose_steps,loss_val,p)
      count::Int += 1
      if mod(count,verbose_steps) == 0
        println("Iteration: $count")
        println("Current Cost: $loss_val")
        println("Parameters: $p")
      end
    end
  function cost_function(p)
    # generate new DEProblem
    tmp_prob = prob_generator(prob,p)
    if typeof(loss) <: DECostFunction
      sol = solve(tmp_prob,alg;saveat=loss.t,save_everystep=false,dense=false,kwargs...)
    else
      sol = solve(tmp_prob,alg;kwargs...)
    end
    if regularization == nothing
      loss_val = loss(sol)
    else
      loss_val = loss(sol) + regularization(p)
    end
    if verbose_opt verbose(count,verbose_steps,loss_val,p) end
    loss_val
  end
  function cost_gradient!(result,p)
    if length(p)>0
      if mpg_autodiff
        gcfg = ForwardDiff.GradientConfig(cost_function, autodiff_prototype, autodiff_chunk)
        ForwardDiff.gradient!(result, cost_function, p, gcfg)
      else
        Calculus.finite_difference!(cost_function,p,result,:central)
      end
    end
    # return cost function result for NLopt
    cost_function(result)
  end
  # number of shooting (subintervals) > 1, using multiple shooting method
  if shooting > 1
    # array of loss subintervals
    loss_subintervals = loss_split(loss, shooting)
    # return an array of DiffEqObjective instances doing single shooting recursively
    [build_loss_objective(prob,alg,subloss,regularization,
                          mpg_autodiff = mpg_autodiff,
                          verbose_opt = verbose_opt,
                          verbose_steps = verbose_steps,
                          prob_generator = prob_generator,
                          autodiff_prototype = autodiff_prototype,
                          autodiff_chunk = autodiff_chunk,
                          kwargs...) for subloss in loss_subintervals]
  else
    # return a DiffEqObjective instance
    DiffEqObjective(cost_function,cost_gradient!)
  end
end

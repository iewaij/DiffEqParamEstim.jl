export DiffEqObjective, build_loss_objective

struct DiffEqObjective{F,N} <: Function
  cost_function::F # objective function for Optim.jl and cost calulation for NLopt.jl
  nlopt_function::N # objective function for NLopt.jl
  #TODO?: currently DiffEqObjective do not have function explicitly update gradient, should we build this and feed it to pakages like Optim.jl?
end

# build function-like objects
(f::DiffEqObjective)(p) = f.cost_function(p)
(f::DiffEqObjective)(p, grad) = f.nlopt_function(p, grad)

#TODO: build loss_split() to other types of loss
function loss_split{T<:Vector, W<:Union{Vector, Void}}(loss::L2Loss{T, T, W}, shooting)
  # missing values error
  @assert length(loss.t) == length(loss.data) "time and data length mismatched"
  len = length(loss.t)
  # get subinterval length
  sub_len = len ÷ shooting
  # split loss instance into equal size subintervals, remain data points would be dropped
  # return an array of loss subintervals
  if loss.weight == nothing
    [L2Loss(loss.t[i:i+sub_len-1], loss.data[i:i+sub_len-1]) for i in 1:sub_len:len-sub_len]
  else
    [L2Loss(loss.t[i:i+sub_len-1], loss.data[i:i+sub_len-1], loss.weight[i:i+sub_len-1]) for i in 1:sub_len:len-sub_len]
  end
end

function loss_split{T<:Vector, M<:Matrix, W<:Union{Vector, Void}}(loss::L2Loss{T, M, W}, shooting)
  # missing values error
  @assert length(loss.t) == size(loss.data)[2] "time and data length mismatched"
  len = length(loss.t)
  # get subinterval length
  sub_len = len ÷ shooting
  # split loss instance into *equal* size subintervals, remain data points would be dropped
  # return an array of loss subintervals
  if loss.weight == nothing
    [L2Loss(loss.t[i:i+sub_len-1], loss.data[:,i:i+sub_len-1]) for i in 1:sub_len:len-sub_len]
  else
    [L2Loss(loss.t[i:i+sub_len-1], loss.data[:,i:i+sub_len-1], loss.weight[i:i+sub_len-1]) for i in 1:sub_len:len-sub_len]
  end
end

function build_loss_objective(prob::DEProblem,alg,loss,regularization=nothing;
                              mpg_autodiff = false,
                              verbose_opt = false,
                              verbose_steps = 100,
                              prob_generator = problem_new_parameters,
                              autodiff_prototype = mpg_autodiff ? zeros(prob.p) : nothing,
                              autodiff_chunk = mpg_autodiff ? ForwardDiff.Chunk(autodiff_prototype) : nothing,
                              kwargs...)
  # keep track of # function evaluations
  count = 0
  function verbose(verbose_steps,p)
    count::Int += 1
    if mod(count,verbose_steps) == 0
      cost = cost_function(p)
      println("Iteration: $count")
      println("Parameters: $p")
      println("Current Cost: $cost")
    end
  end
  function cost_function(p, loss::DECostFunction = loss)
    @assert rem(length(p), length(prob.p)) == 0 "initial values and prblem parameters length mismatched"
    @assert length(p) >= length(prob.p) "too many initial values (subintervals)"
    # number of shooting will be determined by initial values
    shooting = length(p) ÷ length(prob.p)
    if verbose_opt verbose(verbose_steps,p) end
    if shooting == 1
      # generate new temporary DEProblem
      tmp_prob = prob_generator(prob,p)
      if typeof(loss) <: DECostFunction
      ## TODO: Regularization should not be DECostFunction type
        sol = solve(tmp_prob,alg;saveat=loss.t,save_everystep=false,dense=false,kwargs...)
      else
        sol = solve(tmp_prob,alg;kwargs...)
      end
      if regularization == nothing
        loss_val = loss(sol)
      else
        loss_val = loss(sol) + regularization(p)
      end
      loss_val
    # number of shooting (subintervals) > 1, using multiple shooting method
    elseif shooting > 1
      # array of loss subintervals, currently only works for L2Loss TODO: support all DECostFunction
      loss_subintervals = loss_split(loss, shooting)
      ∑cost_function(p, shooting, loss_subintervals)
    end
  end
  function ∑cost_function(p::Vector, shooting, loss_subintervals)
    sum(cost_function(p[i], loss_subintervals[i]) for i in 1:shooting)
  end
  function ∑cost_function(p::Matrix, shooting, loss_subintervals)
    sum(cost_function(view(m, i, :), loss_subintervals[i]) for i in 1:shooting)
  end
  function nlopt_function(p, grad)
    if length(grad)>0
      if mpg_autodiff
        gcfg = ForwardDiff.GradientConfig(cost_function, autodiff_prototype, autodiff_chunk)
        ForwardDiff.gradient!(p, cost_function, grad, gcfg)
      else
        Calculus.finite_difference!(cost_function,grad,p,:central)
      end
    end
    # return cost for NLopt
    cost_function(p)
  end
  # return a DiffEqObjective instance
  DiffEqObjective(cost_function,nlopt_function)
end

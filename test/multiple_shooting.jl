using DifferentialEquations

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

function build_loss_objective_alpha(prob::DEProblem,alg,loss,regularization=nothing;
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
      println("Iteration: $count")
      println("Parameters: $p")
      cost = cost_function(p)
      println("Current Cost: $cost")
    end
  end
  function cost_function(p, loss::DECostFunction = loss)
    @assert rem(length(p), length(prob.p)) == 0 "initial values and prblem parameters length mismatched"
    shooting = length(p) ÷ length(prob.p)
    if verbose_opt verbose(verbose_steps,p) end
    if shooting == 1
      # generate new temporary DEProblem
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

f = @ode_def_nohes LotkaVolterraTest begin
  dx = a*x - x*y
  dy = -3y + x*y
end a

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [2.0] # true parameter is 2.0
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob,Tsit5())

t = collect(linspace(0,10,200)) # 200 data in total
using RecursiveArrayTools
randomized = VectorOfArray([(sol(t[i]) + .09randn(2)) for i in 1:length(t)]) # add normal noise μ = 0 and σ = 0.3
data = convert(Array,randomized)

obj = build_loss_objective_alpha(prob,Tsit5(),L2Loss(t,data), mpg_autodiff = false, verbose_opt = true, verbose_steps = 1, maxiters=10000)

mpg_obj = build_loss_objective_alpha(prob,Tsit5(),L2Loss(t,data), mpg_autodiff = false, verbose_opt = true, verbose_steps = 1, maxiters=10000)

range = 0.0:0.1:10.0
using Plots; plotlyjs()
plot(range,[obj(i) for i in range],yscale=:log10,
     xaxis = "Parameter", yaxis = "Cost", title = "Cost",
     lw = 3, label="Cost Function")

loss = L2Loss(t,data)
sub = loss_split(loss, 4)
sub[1]
[sub[i] for i in 1:2]

obj.cost_function([0.0, 0.0, 3.0])

import Optim
Optim.optimize(obj, [0.0, 0.0, 3.0])

import NLopt
opt = NLopt.Opt(:GN_ESCH, 3)
NLopt.min_objective!(opt, multi_mpg)
NLopt.lower_bounds!(opt,[0.0])
NLopt.upper_bounds!(opt,[5.0])
NLopt.xtol_rel!(opt,1e-3)
NLopt.maxeval!(opt, 100000)
(minf,minx,ret) = NLopt.optimize(opt,[1.0, 1.5, 3.0])

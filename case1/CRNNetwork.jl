using Catalyst,OrdinaryDiffEq,Random
using IterTools: ncycle
using Flux
using Flux.Data: DataLoader
using Flux: mse, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw

rn = @reaction_network begin
    k1, 2*x1 --> x2
    k2, x1 --> x3
    k3, x3 --> x4
    k4, x2 + x4 --> x5
end k1 k2 k3 k4
ns = numspecies(rn); 	nr = numreactions(rn)

# W_sub a matrix of stoichiometric coeffs of reactants
#= one may use to force the input layers weights for faster convergence

W_sub = zeros(Float32,nr, ns)
smap = speciesmap(rn)
for (k,rx) in enumerate(reactions(rn))
	stoich = rx.substoich
	for (i,sub) in enumerate(rx.substrates)
		W_sub[k,smap[sub.op]] = stoich[i]
	end
end=#

n_exp = 30
k = Float32[0.1, 0.2, 0.13, 0.3]
datasize = 200;
tspan = Float32[0.0, 20.0]
tsteps = range(tspan[1], tspan[2], length = datasize)
x0_list = rand(Float32, (n_exp, ns));
x0_list[:, 3:ns] .= 0.

alg = Tsit5()      # ode solver
x = zeros(Float32, ns, datasize*n_exp);
Dx = zeros(Float32, ns, datasize*n_exp);
for i in 1:n_exp
    x0 = x0_list[i, :];     dx = copy(x0);
    prob_trueode = ODEProblem(rn, x0, tspan, k)
    ode_data = Array(solve(prob_trueode, alg, saveat = tsteps))

    D_ode_data = similar(ode_data)
     # ideal-derivative from the model itself
    for j in 1:datasize
        prob_trueode.f(dx, ode_data[:,j], k,tsteps[j])
        D_ode_data[:,j] =  dx
    end

	x[:, (i-1)*datasize + 1 : i*datasize] = ode_data
	Dx[:, (i-1)*datasize + 1 : i*datasize] =  D_ode_data

end

## training data-set 67%,..testing data-set 33%
xtrain = log.(x[:,1:4020] .+ eps(Float32)) ;
xtest = log.(x[:,4021:end] .+ eps(Float32));

Dxtrain = Dx[:,1:4020];
Dxtest = Dx[:,4021:end];

## Training a CRNN
@with_kw mutable struct Args
    η::Float64 = 0.0005          # learning rate
    batchsize::Int = 60        # batch size
    epochs::Int = 10000        # number of epochs
    device::Function = cpu
    # set as gpu, if gpu available
end

function loss_all(dataloader, CRNN)
    l = 0f0
    for (x,y) in dataloader
        l += mse(CRNN(x), y)
    end
    l/length(dataloader)
end

function accuracy(data_loader, CRNN)
    acc = 0
    for (x,y) in data_loader
        acc += mse(CRNN(x),y)
    end
    1 - sqrt(acc/length(data_loader))  # 1 - rmse
end

## Initializing Model parameters
args = Args()
# Batching
train_data = DataLoader(xtrain, Dxtrain, batchsize = args.batchsize,
 				shuffle = true)
test_data = DataLoader(xtest, Dxtest, batchsize = args.batchsize)

# Construct model... chemical reaction Neural net
CRNN = Chain( Dense(ns,nr,exp) , Dense(nr,ns) )
ps = Flux.params(CRNN)  		# new params
delete!(ps, CRNN[2].b) 			# Do not optimise bias of Layer 2

train_data = args.device.(train_data)
test_data = args.device.(test_data)
CRNN = args.device(CRNN)
loss(x,y) = mse(CRNN(x), y)

evalcb = () -> @show(loss_all(train_data, CRNN))
opt = ADAM(args.η)

@epochs args.epochs Flux.train!( loss, ps, train_data,
		opt, cb = throttle(evalcb ,5) )

@show accuracy(train_data, CRNN)
@show accuracy(test_data, CRNN)

in_wt = CRNN[1].W
out_wt =  zeros(Float32, ns, nr)
k_wt = zeros(Float32,nr,1)
# scaling to find out_wt

for i in 1:nr		# out the
	out_wt[:,i] = (CRNN[2].W[:,i])./maximum(CRNN[2].W[:,i])
	k_wt[i,1] = (exp.(CRNN[1].b).*maximum(abs.(CRNN[2].W[:,i])))[i,1]
end

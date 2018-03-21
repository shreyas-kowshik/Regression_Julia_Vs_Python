# Pkg.add("Plots")
# pyplot()

tic()

reg = 0.5
alpha = 0.1

function loss(y_,y,W)
	err = y_-y
	sq_err = err.*err
	loss = sum(sq_err)

	N = size(y,1)
	loss = loss/(2N)

	W[1,1] = 0
	W = W.*W
	loss+=(reg*sum(W))

	return loss
end

N = 10000
D = 300
W = randn(D+1,1)
one = ones(N,1)
X = randn(N,D)
X = [one X]
y = randn(N,1)
y_ = X*W

init_loss = loss(y_,y,W)
println("Initial loss : ",init_loss)
#D+1,1 X=N,D+1 y_-y=N,1
for i in 1:1000
	y_ = X*W
	#println(loss(y_,y,W))
	c = (1-(alpha*reg/N))
	W = W*c - (alpha/N)*(X'*(y_-y))
end

final_loss = loss(y_,y,W)
println("Final loss : ",final_loss)

println("Loss Ratio : ",init_loss/final_loss)
# X = [1 2 3;4 5 6;6 5 3]
# y = [1;2;3]
# print(X*y)
toc()

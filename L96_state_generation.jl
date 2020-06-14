using LinearAlgebra
using Statistics
using DelimitedFiles
using Random


##L96modelの右辺
function L96(u;F=8.0,N=40)
    f = fill(0.0, N)
    for k in 3:N-1
        f[k] = (u[k+1]-u[k-2])u[k-1] - u[k] + F
    end
    f[1] = (u[2]-u[N-1])u[N] - u[1] + F
    f[2] = (u[3]-u[N])u[1] - u[2] + F
    f[N] =  (u[1]-u[N-2])u[N-1] - u[N] + F

    return f
end

#4-4Runge-Kutta
function Model(u;dt=0.05)
    du = u
    s1 = L96(u .+ dt)
    s2 = L96(u + s1*dt/2)
    s3 = L96(u + s2*dt/2)
    s4 = L96(u + s3*dt)
    du += (s1 + 2*s2 + 2*s3 + s4)*(dt/6)
    return du
end

function main()
Time_Step = 14600
F = 8.0
N = 40
u = fill(F,N) + rand(N)

##一年分はスピンアップとして捨てる
for i in 1:Time_Step
    u = Model(u)
end

##一年分を真値として保存
open("L96_truestate.txt","w") do truestate
    for i in 1:Time_Step
        u = Model(u)
        writedlm(truestate, [(i/40) u'])
    end
end

##真値にノイズを足して観測データを作る
u_true = readdlm("L96_truestate.txt")

open("L96_observation.txt","w") do observation
    for i in 1:Time_Step
        writedlm(observation, [(i/40) (u_true[i,2:N+1]+randn(N))'])
    end
end
end

main()
##ちゃんとできているかチェック
u_obs = readdlm("L96_observation.txt")
RMSE = fill(0.0, Time_Step)
for i in 1:Time_Step
    global RMSE[i] = norm(u_true[i,2:N+1]-u_obs[i,2:N+1])/sqrt(N)
end

plot(1:Time_Step, RMSE[1:Time_Step],label="RMSE",lw=2)

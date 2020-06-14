using LinearAlgebra
using Statistics
using Random
using DelimitedFiles


##L96modelの右辺
function L96(u; F = 8.0, N = 40)
    f = fill(0.0, N)
    for k = 3:N-1
        f[k] = (u[k+1] - u[k-2]) * u[k-1] - u[k] + F
    end
    f[1] = (u[2] - u[N-1]) * u[N] - u[1] + F
    f[2] = (u[3] - u[N]) * u[1] - u[2] + F
    f[N] = (u[1] - u[N-2]) * u[N-1] - u[N] + F

    return f
end

#4-4Runge-Kutta
function Model(u; dt = 0.05)
    du = u
    s1 = L96(u .+ dt)
    s2 = L96(u + s1 * dt / 2)
    s3 = L96(u + s2 * dt / 2)
    s4 = L96(u + s3 * dt)
    du += (s1 + 2 * s2 + 2 * s3 + s4) * (dt / 6)
    return du
end

function localization(i,j;N = 40, d = 25.0)
    dist = min(abs(i-j), abs(abs(i-j)-N))
    return exp(-dist^2 / d)
end

function main()
    Time_Step = 14600
    member = 40
    N = 40
    M = 40
    F = 8.0
    IN = Matrix(1.0I, N, N)
    delta = 1.0e-5

    u_true = readdlm("L96_truestate.txt")
    u_obs = readdlm("L96_observation.txt")


    H = Matrix(1.0I, M, N)
    R = Matrix(1.0I, M, M)
    y = fill(0.0, M)

    ua = fill(0.0, (N,member))
    uf_mean = fill(0.0, N)
    ua_mean = fill(0.0, N)
    uf = fill(0.0, (N,member))
    dXf = fill(0.0, (N,member))
    dYf = fill(0.0, (M, member))
    dXa = fill(0.0, (N,member))
    localization_mat = fill(0.0, (N,N))
    for i in 1:N
        for j in 1:N
            localization_mat[i,j] = localization(i,j)
        end
    end

    open("L96_EnKF_Serial_output.txt","w") do output
        for m = 1:member
            ua[:, m] = rand(N) .+ F
            for i = 1:Time_Step
                ua[:,m] = Model(view(ua,:,m))
            end
        end

        for i in 1:Time_Step
            y = u_obs[i, 2:N+1]
            ##forecast step
            for m in 1:member
                uf[:,m] = Model(view(ua,:,m))
            end
            uf_mean = mean(uf, dims = 2)
            dXf = uf .- uf_mean
            Pf =  1.1 * dXf * dXf' / (member - 1)
            Pf = localization_mat .* Pf

            ##analysis step
            dXa = dXf
            Pa = Pf
            ua_mean = uf_mean

            for k in 1:M
                H_serial = fill(0.0, M)
                H_serial[k] = 1.0
                K = Pa[:,k] / (1.0 + Pa[k,k])
                K_serial = K / (1.0 + (1.0 / sqrt(1.0+Pa[k,k])))

                ua_mean = ua_mean + K * (y[k] - ua_mean[k])
                dXa = (I - K_serial*H_serial')* dXa
                Pa = dXa * dXa' / (member - 1.0)
                Pa = localization_mat .* Pa
            end

            ua = dXa .+ ua_mean

            writedlm(output,[(i / 40) (norm(u_true[i, 2:N+1] - ua_mean)/sqrt(N)) (norm(u_obs[i, 2:N+1] -u_true[i, 2:N+1])/sqrt(N))])
        end
    end
end

main()

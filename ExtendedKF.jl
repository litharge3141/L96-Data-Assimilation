using LinearAlgebra
using Statistics
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

function main()
    Time_Step = 14600
    N = 40
    M = 40
    F = 8.0
    IN = Matrix(1.0I, N, N)
    delta = 1.0e-5

    u_true = readdlm("L96_truestate.txt")
    u_obs = readdlm("L96_observation.txt")


    H = Matrix(1.0I, M, M)
    R = Matrix(1.0I, M, M)

    open("L96_EKF_output_noinflation.txt", "w") do output
        ua = rand(N) .+ F
        for i = 1:Time_Step
            ua = Model(ua)
        end
        Pa = 25.0 * IN

        for i = 1:Time_Step
    ##forecast step
            uf = Model(ua)
            JM = zeros(N, N)
            for j = 1:M
                JM[:, j] = (Model(ua + delta * IN[:, j]) - Model(ua)) / delta
            end
            Pf = JM * Pa * JM'

    ##analysis step
            K = Pf * H' * inv(H * Pf * H' + R)
            ua = uf + K * (u_obs[i, 2:N+1] - H * uf)
            Pa = (I - K * H)Pf

    ##output
            writedlm(
                output,
                [(i / 40) sqrt(tr(Pa) / N) (norm(u_true[i, 2:N+1] - uf) /
                                            sqrt(N)) (norm(u_obs[i, 2:N+1] -
                                                           u_true[i, 2:N+1]) /
                                                      sqrt(N))]
            )
        end
    end
end

main()

using Plots
using DifferentialEquations

function run()
	include("sho.jl")
end

mutable struct Particle
	x::Vector{Float64}
	v::Vector{Float64}
end

mutable struct Manifold
	dt::Float64
	tf::Float64
	nt::Int64		# num time steps
	t::Float64
	tpoints::Vector{Float64}
	ω0::Float64
	ω::Float64
	b::Float64
	A::Float64
	particle::Particle
end

function initialize!(;dt::Float64, tf::Float64, ω0::Float64, ω::Float64, 
		b::Float64, A::Float64, x0::Float64, v0::Float64)
	nt = Int64(tf/dt)
	particle = Particle(zeros(nt), zeros(nt))
	particle.x[1] = x0
	particle.v[1] = v0
	t = 0.0
	tpoints = 0.0:dt:(tf-dt)
	return Manifold(dt, tf, nt, t, tpoints, ω0, ω, b, A, particle)
end

function compute_acceleration(t::Int64, m::Manifold)
	k = (m.ω0)^2
	a = -k * m.particle.x[t] - 
		m.b * m.particle.v[t] - 
		0.0 * m.particle.x[t]^3 + 
		m.A * cos(m.ω * m.t)
	return a
end

function integrate!(m::Manifold)
	# manually compute t = 0 case
	t = 1
	a = compute_acceleration(t, m)
	m.particle.v[2] = m.particle.v[1] + a * m.dt * 0.5
	m.particle.x[2] = m.particle.x[1] + m.particle.v[2] * m.dt * 0.5
	m.t += m.dt

	for t in 2:(m.nt-1)
		a = compute_acceleration(t, m)
		m.particle.v[t+1] = m.particle.v[t] + a * m.dt
		m.particle.x[t+1] = m.particle.x[t] + m.particle.v[t+1] * m.dt
		m.t += m.dt
	end
end

function tendency!(dΓ::Vector{Float64}, Γ::Vector{Float64}, p, t::Float64)
	x = Γ[1]
	v = Γ[2]
	k = (m.ω0)^2
	a = -k * x - m.b * v - 0.0 * x^3 + m.A * cos(m.ω * t)
	dΓ[1] = v
	dΓ[2] = a
end

function run_solver()
	Γ0 = [m.particle.x[1], m.particle.v[1]]
	tspan = (0.0, m.tf)
	prob = ODEProblem(tendency!, Γ0, tspan)
	sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
	xtplot = plot(sol, idxs = (0,1))
	xvplot = plot(sol, idxs = (1,2))
	plot(xtplot, xvplot, size=(600,300))
end

function run_integrator()
	integrate!(m)
	xtplot = plot(m.tpoints, m.particle.x)
	xvplot = plot(m.particle.x, m.particle.v)
	plot(xtplot, xvplot, size=(600,300))
end

m = initialize!(
	dt = 0.001, 
	tf = 100.0, 
	ω0 = 1.0, 
	ω = 1.2, 
	b = 0.1, 
	A = 0.5,
	x0 = -3.0, 
	v0 = 0.0
	)

run_integrator()
#run_solver()
